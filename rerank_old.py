from query import Query
from retrieve import RetrievedInfo
from constants import CHECK_NUM, MIN_CHECK_NUM, MAX_CHECK_NUM
import numpy as np
from copy import deepcopy
import xgboost as xgb
import time
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import cvxpy as cp
from scipy.optimize import minimize
from collections import defaultdict


def weighted_logistic_loss(w, X, y, pos_weight=10, neg_weight=1):
    z = np.dot(X, w)
    weights = pos_weight * y + neg_weight * (1 - y)
    y = 2 * y - 1
    weighted_loss = np.mean(weights * np.log(1 + np.exp(-y * (z))))
    # coef_diff_loss = (w[1] + w[2] - w[3] - w[4]) ** 2
    reg_loss = 1e-3 * np.sum(w[1:] ** 2)
    loss = weighted_loss + reg_loss
    return loss


def pred_logistic(X, w):
    X = np.c_[np.ones(X.shape[0]), X]
    z = np.dot(X, w)
    return 1 / (1 + np.exp(-z))


def prepare_data(
    query: Query, bm25_objs, hnsw_objs, bm25_obj_score_dict, hnsw_obj_score_dict
):
    pos_samples, neg_samples = [], []
    pos_objs, neg_objs = [], []
    max_bm25_score = 0
    for i, (obj, score) in enumerate(query.obj_scores.items()):
        feature = query.obj_features[obj]
        max_bm25_score = max(max_bm25_score, feature[0])

    for i, (obj, score) in enumerate(query.obj_scores.items()):
        feature = deepcopy(query.obj_features[obj])
        # feature[0] = feature[0] / max_bm25_score
        # feature.append(max(feature[0], feature[1]))
        if score > 0:
            pos_samples.append(feature)
            pos_objs.append(obj)
        else:
            neg_samples.append(feature)
            neg_objs.append(obj)
    train_X = np.concatenate([pos_samples, neg_samples], axis=0)
    train_y = np.concatenate(
        [np.ones(len(pos_samples)), np.zeros(len(neg_samples))], axis=0
    )

    test_X = []
    candidate_objs = list(
        (set(bm25_objs) | set(hnsw_objs)) - set(list(query.obj_scores.keys()))
    )
    print(len(candidate_objs))
    for i, obj in enumerate(candidate_objs):
        bm25_score = bm25_obj_score_dict.get(obj, [0, 0])
        hnsw_score = hnsw_obj_score_dict.get(obj, [0, 0])
        feature = [bm25_score[0], bm25_score[1], hnsw_score[0], hnsw_score[1]]
        test_X.append(feature)
    test_X = np.array(test_X)
    return train_X, train_y, test_X, candidate_objs


def process_feature(train_X, test_X):
    def normalize(X, max_scalar_v1, max_scalar_v3):
        X[:, 0] = X[:, 0] / max_scalar_v1
        X[:, 1] = X[:, 1] / max_scalar_v1
        X[:, 2] = X[:, 2] / max_scalar_v3
        X[:, 3] = X[:, 3] / max_scalar_v3
        # new_feature_1 = np.maximum(X[:, 0], X[:, 2])
        # new_feature_2 = np.maximum(X[:, 1], X[:, 3])
        # X = np.c_[X, new_feature_1, new_feature_2]
        return X

    max_scalar_v1 = np.max(train_X[:, 0])
    max_scalar_v3 = np.max(train_X[:, 2])
    print(f"max_scalar_v1: {max_scalar_v1}, max_scalar_v3: {max_scalar_v3}")
    train_X = normalize(train_X, max_scalar_v1, max_scalar_v3)
    test_X = normalize(test_X, max_scalar_v1, max_scalar_v3)
    return train_X, test_X


def agg_feature(train_X, test_X, bm25_weights, hnsw_weights):
    def weighted_agg(X, bm25_weights, hnsw_weights):
        bm25_feature = X[:, 0] * bm25_weights[0] + X[:, 1] * bm25_weights[1]
        hnsw_feature = X[:, 2] * hnsw_weights[0] + X[:, 3] * hnsw_weights[1]
        return np.c_[bm25_feature, hnsw_feature]

    def normalize(X, max_bm25, max_hnsw):
        X[:, 0] = X[:, 0] / max_bm25
        X[:, 1] = X[:, 1] / max_hnsw
        return X

    train_X = weighted_agg(train_X, bm25_weights, hnsw_weights)
    test_X = weighted_agg(test_X, bm25_weights, hnsw_weights)
    max_bm25 = max(np.max(train_X[:, 0]), 1e-6)
    max_hnsw = max(np.max(train_X[:, 1]), 1e-6)
    train_X = normalize(train_X, max_bm25, max_hnsw)
    test_X = normalize(test_X, max_bm25, max_hnsw)
    return train_X, test_X


def split_data(train_X, train_y):
    # split the data based on whether bm25=0
    train_X_p1 = train_X[(train_X[:, 0] > 0) | (train_y == 1)]
    train_X_p2 = train_X[(train_X[:, 0] == 0) & (train_y == 0)]
    train_y_p1 = train_y[(train_X[:, 0] > 0) | (train_y == 1)]
    train_y_p2 = train_y[(train_X[:, 0] == 0) & (train_y == 0)]
    np.random.shuffle(train_X_p2)
    train_X_p1 = np.concatenate([train_X_p1, train_X_p2[: len(train_X_p1)]], axis=0)
    train_y_p1 = np.concatenate([train_y_p1, train_y_p2[: len(train_y_p1)]], axis=0)

    return (train_X_p1, train_y_p1, train_X, train_y)


def determine_threshold(train_y_prob, train_y):
    train_y_prob_pos = sorted(train_y_prob[train_y == 1], reverse=True)
    y_prob_thr = min(train_y_prob_pos)
    print(f"y_prob_thr: {y_prob_thr:.4f}")
    y_prob_thr = y_prob_thr - np.std(train_y_prob[train_y_prob >= y_prob_thr]) * 1.96
    print(f"y_prob_thr: {y_prob_thr:.4f}")
    y_prob_thr = min(y_prob_thr, 0.5)
    print(f"y_prob_thr: {y_prob_thr:.4f}")
    return y_prob_thr


def get_next_objs(
    bm25_objs, bm25_agg_pos_scores, hnsw_objs, hnsw_agg_pos_scores, query: Query
):
    # sort the retrieved objs by the maximum of bm25 and hnsw scores
    obj_to_check_scores = defaultdict(int)
    bm25_obj_score_dict = defaultdict(int)
    hnsw_obj_score_dict = defaultdict(int)
    for i, (obj, score) in enumerate(zip(bm25_objs, bm25_agg_pos_scores)):
        bm25_obj_score_dict[obj] = score
    for i, (obj, score) in enumerate(zip(hnsw_objs, hnsw_agg_pos_scores)):
        hnsw_obj_score_dict[obj] = score
    for obj in set(bm25_objs) | set(hnsw_objs):
        score1 = bm25_obj_score_dict.get(obj, 0)
        score2 = hnsw_obj_score_dict.get(obj, 0)
        obj_to_check_scores[obj] = max(score1, score2)  #  + (score1 + score2) / 2
    for obj, score in query.obj_scores.items():
        try:
            obj_to_check_scores[obj] = -1
        except:
            pass
    # sort the objs by the scores
    obj_to_check_scores = sorted(
        obj_to_check_scores.items(), key=lambda x: x[1], reverse=True
    )
    # get the objs to check
    obj_to_check = [obj for obj, _ in obj_to_check_scores[:CHECK_NUM]]
    bm25_positions = [
        bm25_objs.index(obj) if obj in bm25_objs else -1 for obj in obj_to_check
    ]
    bm25_scores = [bm25_agg_pos_scores[i] if i != -1 else 0 for i in bm25_positions]
    hnsw_positions = [
        hnsw_objs.index(obj) if obj in hnsw_objs else -1 for obj in obj_to_check
    ]
    hnsw_scores = [hnsw_agg_pos_scores[i] if i != -1 else 0 for i in hnsw_positions]
    print(f"bm25_positions: {bm25_positions}")
    print(f"bm25_scores: {list(np.round(bm25_scores, 4))}")
    print(f"hnsw_positions: {hnsw_positions}")
    print(f"hnsw_scores: {list(np.round(hnsw_scores, 4))}")
    return obj_to_check


def learn_weights(train_X, train_y, name="bm25"):
    train_X_with_intercept = np.c_[np.ones(train_X.shape[0]), train_X]
    w_init = np.zeros(train_X_with_intercept.shape[1])
    constraints = [{"type": "ineq", "fun": lambda w: w[1:]}]
    if name == "all":
        # # max > bm25, max > hnsw
        # constraints.append({"type": "ineq", "fun": lambda w: w[3] - w[1]})
        # constraints.append({"type": "ineq", "fun": lambda w: w[3] - w[2]})
        # bm25 > hnsw * 0.1, hnsw > bm25 * 0.1
        constraints.append({"type": "ineq", "fun": lambda w: w[2] - w[1] * 0.1})
        constraints.append({"type": "ineq", "fun": lambda w: w[1] - w[2] * 0.1})
    # else:
    #     # max > 0.5 * avg, avg > 0.5 * max
    #     constraints.append({"type": "ineq", "fun": lambda w: w[1] - 0.5 * w[2]})
    #     constraints.append({"type": "ineq", "fun": lambda w: w[2] - 0.5 * w[1]})
    pos_weight = min(max(1, np.sum(train_y == 0) / np.sum(train_y == 1)), 10)
    result = minimize(
        weighted_logistic_loss,
        w_init,
        args=(train_X_with_intercept, train_y, pos_weight, 1),
        method="SLSQP",
        constraints=constraints,
    )
    w_optimized = result.x
    # print(f"{name}_w_optimized: {list(np.round(w_optimized, 4))}")
    return w_optimized


def eval_pred(y_pred, y_true, beta=10):
    pre = np.sum(y_pred * y_true) / max(np.sum(y_pred), 1e-6)
    rec = np.sum(y_pred * y_true) / max(np.sum(y_true), 1e-6)
    f1 = (1 + beta) * pre * rec / (beta * pre + rec + 1e-6)
    return f1


def learn_p1_weights(train_X_org, train_y):
    best_alpha = 1
    best_f1 = 0
    best_weights = None
    for alpha in np.arange(1, 5, 0.1):
        train_X = deepcopy(train_X_org)
        new_feature = np.maximum(train_X[:, 0] * alpha, train_X[:, 1])
        train_X = np.c_[train_X, new_feature]  # [N, 3]
        # learn the weights for bm25, hnsw, and max
        weights = learn_weights(train_X, train_y, name="all")
        y_prob = pred_logistic(train_X, weights)
        y_pred = (y_prob > 0.5).astype(int)
        f1 = eval_pred(y_pred, train_y)
        # print(f"[DEBUG] alpha: {alpha:.4f}, weighted_f1: {f1:.4f}")
        if f1 > best_f1:
            best_alpha = alpha
            best_f1 = f1
            best_weights = deepcopy(weights)
    print(f"best_alpha: {best_alpha:.4f}")
    print(f"best_weights: {list(np.round(best_weights, 4))}")
    return best_alpha, best_weights


def rerank_retrieved_objs(
    query: Query, retrieved_info: RetrievedInfo, args, corpus: list
):
    bm25_objs = retrieved_info.bm25_objs
    hnsw_objs = retrieved_info.hnsw_objs
    bm25_agg_pos_scores = retrieved_info.bm25_agg_pos_scores
    hnsw_agg_pos_scores = retrieved_info.hnsw_agg_pos_scores
    bm25_agg_pos_scores_avg = retrieved_info.bm25_agg_pos_scores_avg
    hnsw_agg_pos_scores_avg = retrieved_info.hnsw_agg_pos_scores_avg
    bm25_agg_pos_scores_max = retrieved_info.bm25_agg_pos_scores_max
    hnsw_agg_pos_scores_max = retrieved_info.hnsw_agg_pos_scores_max
    bm25_obj_score_dict = {
        obj: [max_score, avg_score]
        for obj, max_score, avg_score in zip(
            bm25_objs, bm25_agg_pos_scores_max, bm25_agg_pos_scores_avg
        )
    }
    hnsw_obj_score_dict = {
        obj: [max_score, avg_score]
        for obj, max_score, avg_score in zip(
            hnsw_objs, hnsw_agg_pos_scores_max, hnsw_agg_pos_scores_avg
        )
    }

    pos_num = sum(list(query.obj_scores.values()))
    if query.org_query in query.obj_scores:
        pos_num -= 1
    if pos_num < 2:
        obj_to_check = get_next_objs(
            bm25_objs, bm25_agg_pos_scores, hnsw_objs, hnsw_agg_pos_scores, query
        )
        return obj_to_check, False

    # TODO: train the model to rerank the objs
    train_X, train_y, test_X, candidate_objs = prepare_data(
        query, bm25_objs, hnsw_objs, bm25_obj_score_dict, hnsw_obj_score_dict
    )

    # (bm25_max, bm25_avg, hnsw_max, hnsw_avg) normalization to [0, 1]
    train_X_dim4, test_X_dim4 = process_feature(train_X, test_X)
    # print("sample train_X normalized (bm25_max, bm25_avg, hnsw_max, hnsw_avg)")
    # for x, y in zip(train_X_dim4, train_y):
    #     if y == 1:
    #         print(y, list(np.round(x, 4)))

    bm25_weights = [0.5, 0.5]
    hnsw_weights = [0.5, 0.5]
    train_X_dim2, test_X_dim2 = agg_feature(
        train_X_dim4, test_X_dim4, bm25_weights, hnsw_weights
    )
    print("sample train_X_dim2:")
    neg_cnt = 0
    for x, y in zip(train_X_dim2, train_y):
        if y == 1 or neg_cnt < 5:
            if y == 0:
                neg_cnt += 1
            print(y, list(np.round(x, 4)))

    # split the data based on whether bm25=0
    (train_X_p1, train_y_p1, train_X_p2, train_y_p2) = split_data(train_X_dim2, train_y)
    print(f"train_X_p1 (pos/all): {np.sum(train_y_p1)}/{len(train_y_p1)}")
    print(f"train_X_p2 (pos/all): {np.sum(train_y_p2)}/{len(train_y_p2)}")
    test_X_p1 = deepcopy(test_X_dim2)
    test_X_p2 = deepcopy(test_X_dim2)

    obj_score = defaultdict(float)
    # search the scalar for bm25, and learn the weights for bm25, hnsw, and max
    if len(train_X_p1) > 0:
        # learn the weights for bm25, hnsw, and max
        best_alpha, best_weights = learn_p1_weights(train_X_p1, train_y_p1)
        train_new_feature = np.maximum(train_X_p1[:, 0] * best_alpha, train_X_p1[:, 1])
        train_X_p1 = np.c_[train_X_p1, train_new_feature]
        train_y_p1_prob = pred_logistic(train_X_p1, best_weights)

        # estimate the threshold
        threshold_p1 = determine_threshold(train_y_p1_prob, train_y_p1)

        # predict on the test set
        test_new_feature = np.maximum(test_X_p1[:, 0] * best_alpha, test_X_p1[:, 1])
        test_X_p1 = np.c_[test_X_p1, test_new_feature]
        test_y_prob = pred_logistic(test_X_p1, best_weights)
        for obj, prob in zip(candidate_objs, test_y_prob):
            obj_score[obj] = prob
    else:
        threshold_p1 = 0.5

    # learn the weights for p2
    weights_p2 = learn_weights(train_X_p2[:, 1:], train_y_p2, name="only_hnsw")
    # calculate the threshold for p2
    train_y_p2_prob = pred_logistic(train_X_p2[:, 1:], weights_p2)
    threshold_p2 = determine_threshold(train_y_p2_prob, train_y_p2)
    test_y_prob = pred_logistic(test_X_p2[:, 1:], weights_p2)
    for obj, prob in zip(candidate_objs, test_y_prob):
        obj_score[obj] = max(obj_score.get(obj, 0), prob)

    # sort the objs by the scores
    obj_score = sorted(obj_score.items(), key=lambda x: x[1], reverse=True)

    threshold = min(threshold_p1, threshold_p2)
    # get the objs to check
    obj_to_check = []
    for i, (obj, score) in enumerate(obj_score):
        if (score > threshold or i < MIN_CHECK_NUM) and i < min(
            MAX_CHECK_NUM, args.budget - len(query.obj_scores)
        ):
            obj_to_check.append(obj)
    return obj_to_check, len(obj_to_check) == MAX_CHECK_NUM


def get_next_objs_by_threshold(
    train_X, train_y, bm25_thr, hnsw_thr, candidate_objs, test_X, args, query: Query
):
    # update threshold to threshold - 1.96 * std
    def update_threshold(scores, thr):
        scores = scores[scores >= thr]
        std = np.std(scores) if len(scores) > 1 else 0
        thr = thr - std * 1.96
        return thr

    bm25_scores = train_X[:, 0]
    bm25_thr = update_threshold(bm25_scores, bm25_thr)
    hnsw_scores = train_X[:, 1]
    hnsw_thr = update_threshold(hnsw_scores, hnsw_thr)
    print(f"bm25_thr: {bm25_thr:.4f}, hnsw_thr: {hnsw_thr:.4f}")

    obj_score = {}
    for obj, x in zip(candidate_objs, test_X):
        score = max(x[0] / bm25_thr, x[1] / hnsw_thr)
        obj_score[obj] = score
    obj_score = sorted(obj_score.items(), key=lambda x: x[1], reverse=True)
    obj_to_check = []
    for i, (obj, score) in enumerate(obj_score):
        if (score >= 1 or i < MIN_CHECK_NUM) and i < min(
            MAX_CHECK_NUM, args.budget - len(query.obj_scores)
        ):
            obj_to_check.append(obj)
    return obj_to_check
