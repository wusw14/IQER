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
    coef_diff_loss = (w[1] + w[2] - w[3] - w[4]) ** 2
    loss = weighted_loss + 1e-3 * (np.sum(w[1:] ** 2) + coef_diff_loss)
    return loss


def pred_logistic(X, w):
    z = np.dot(X, w)
    return 1 / (1 + np.exp(-z))


def prepare_data(query: Query):
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
    dev_X = np.concatenate([pos_samples, neg_samples], axis=0)
    dev_objs = pos_objs + neg_objs
    dev_y = np.concatenate(
        [np.ones(len(pos_samples)), np.zeros(len(neg_samples))], axis=0
    )
    # create some pos samples
    constructed_num = len(neg_samples) - len(pos_samples)
    constructed_pos_samples = []
    for i in range(constructed_num):
        sample_idx = np.random.randint(0, len(pos_samples))
        sample = pos_samples[sample_idx]
        sample = deepcopy(sample)
        idx = np.random.randint(0, 2)
        if idx < 2:
            val = sample[idx] + np.random.uniform(0, 0.2)
            val = min(1, val)
        # else:
        #     val = sample[idx] - np.random.uniform(0, 0.2)
        #     val = max(0, val)
        sample[idx] = val
        sample[2] = max(sample[0], sample[1])
        constructed_pos_samples.append(sample)
    pos_samples.extend(constructed_pos_samples)
    train_X = np.concatenate([pos_samples, neg_samples], axis=0)
    train_y = np.concatenate(
        [np.ones(len(pos_samples)), np.zeros(len(neg_samples))], axis=0
    )
    return train_X, train_y, dev_X, dev_y, dev_objs


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
        return obj_to_check, False

    # TODO: train the model to rerank the objs
    train_X, train_y, dev_X, dev_y, dev_objs = prepare_data(query)
    print(f"sample train_X: {(np.round(train_X[:3], 4)).tolist()}")
    # data normalization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(train_X)
    train_X_with_intercept = np.c_[np.ones(train_X.shape[0]), X_scaled]
    dev_X_with_intercept = np.c_[np.ones(dev_X.shape[0]), scaler.transform(dev_X)]
    w_init = np.zeros(train_X_with_intercept.shape[1])
    constraints = [{"type": "ineq", "fun": lambda w: w[1:]}]
    result = minimize(
        weighted_logistic_loss,
        w_init,
        args=(train_X_with_intercept, train_y, 10, 1),
        method="SLSQP",
        constraints=constraints,
    )
    w_optimized = result.x
    print(f"w_optimized: {w_optimized}")
    y_prob = pred_logistic(dev_X_with_intercept, w_optimized)
    y_pred = (y_prob > 0.5).astype(int)
    print("\nclassification report:\n", classification_report(dev_y, y_pred))
    # threshold: minimum of y_prob where train_y is 1
    y_prob_pos = sorted(y_prob[dev_y == 1], reverse=True)
    thr = min(y_prob_pos)
    print(f"y_prob: {list(np.round(y_prob[y_prob >= thr], 4))}")
    thr = thr - np.std(y_prob[y_prob >= thr])
    print(f"threshold: {thr}")
    thr = min(thr, 0.5)

    # predict
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
    print(np.mean(test_X, axis=0))
    print(np.std(test_X, axis=0))
    test_X = np.c_[np.ones(test_X.shape[0]), scaler.transform(test_X)]
    # y_pred_prob = clf.predict_proba(test_X)[:, 1]
    y_pred_prob = pred_logistic(test_X, w_optimized)
    # test_X = scaler.transform(test_X)
    # y_pred_prob = calibrated_svm.predict_proba(test_X)[:, 1]
    # dtest = xgb.DMatrix(test_X)
    # y_pred_prob = model.predict(dtest)
    y_pred_prob = np.array(y_pred_prob)
    indices = np.argsort(y_pred_prob)[::-1]
    obj_to_check = []
    for i, idx in enumerate(indices):
        if (y_pred_prob[idx] > thr or i < MIN_CHECK_NUM) and i < min(
            MAX_CHECK_NUM, args.budget - len(query.obj_scores)
        ):
            obj_to_check.append(candidate_objs[idx])
    if len(obj_to_check) == MAX_CHECK_NUM:
        print("[DEBUG]")
        for x, y, y_true, obj in zip(dev_X, y_prob, dev_y, dev_objs):
            print(obj, y_true, y, list(np.round(x, 4)))
    return obj_to_check, len(obj_to_check) == MAX_CHECK_NUM
