from query import Query
from retrieve import RetrievedInfo
from constants import CHECK_NUM, MIN_CHECK_NUM, MAX_CHECK_NUM
import numpy as np
from copy import deepcopy
from collections import defaultdict
import json


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
    print(f"pos_objs: {pos_objs}")
    print(f"neg_objs: {neg_objs[:5]}")
    train_X = np.concatenate([pos_samples, neg_samples], axis=0)
    train_objs = pos_objs + neg_objs
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
    return train_X, train_y, test_X, candidate_objs, train_objs


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
    return train_X, test_X  # [N1, 2], [N2, 2]


def search_best_params(X, y):
    # w1 * bm25 + w2 * hnsw + (1 - w1 - w2) * max(bm25, hnsw)
    # best_ndcg = 0
    best_metric = len(y)
    best_w1, best_w2 = 0, 0
    for w1 in np.arange(0, 1, 0.1):
        for w2 in np.arange(0, 1, 0.1):
            if w1 + w2 > 1:
                continue
            w3 = 1 - w1 - w2
            score = w1 * X[:, 0] + w2 * X[:, 1] + w3 * np.maximum(X[:, 0], X[:, 1])
            _, sorted_y = zip(*sorted(zip(score, y), key=lambda x: x[0], reverse=True))
            # calculate ndcg
            dcg = 0
            for i, y_i in enumerate(sorted_y):
                if y_i == 1:
                    dcg += 1 / np.log2(i + 2)
                    k = i + 1
            idcg = sum([1 / np.log2(i + 2) for i in range(int(sum(y)))])
            ndcg = dcg / idcg
            metric_value = k - ndcg
            if metric_value < best_metric or (
                metric_value == best_metric
                and w1 + w2 < best_w1 + best_w2
                and w1 + w2 > 0
            ):
                best_ndcg = ndcg
                best_k = k
                best_metric = metric_value
                best_w1, best_w2 = w1, w2
                best_score = deepcopy(score)
    neg_cnt = 0
    for i, score in enumerate(best_score):
        if y[i] == 1 or neg_cnt < 5:
            print(f"y={y[i]}, score={score:.2f}, feature={list(np.round(X[i], 4))}")
        if y[i] == 0:
            neg_cnt += 1
    # print(f"BEST NDCG: {best_ndcg:.4f}")
    print(f"BEST K: {best_k}, BEST NDCG: {best_ndcg:.4f}")
    return best_w1, best_w2, best_score


def determine_check_num_and_threshold(train_y, train_score):
    # check number
    sorted_score, sorted_y = zip(
        *sorted(zip(train_score, train_y), key=lambda x: x[0], reverse=True)
    )
    last_pos_position = -1
    last_pos_score = -1
    position_diff_list = []
    score_diff_list = []
    score_diff = None
    score_diff_max = 0.05
    min_pos_score = 1
    for i, (score, y) in enumerate(zip(sorted_score, sorted_y)):
        if y == 1:
            position_diff_list.append(i - last_pos_position)
            last_pos_position = i
            if last_pos_score - score > 0:
                if score_diff is None:
                    score_diff = last_pos_score - score
                else:
                    score_diff = (score_diff + last_pos_score - score) / 2
                score_diff_max = max(score_diff_max, last_pos_score - score)
            score_diff_list.append(max(last_pos_score - score, 0))
            last_pos_score = min(score, 2)
            min_pos_score = min(min_pos_score, score)
    if score_diff is None:
        score_diff = 0
    check_num = np.max(position_diff_list) * 2
    # threshold = min_pos_score - 1.96 * np.std(score_diff_list)
    threshold = min_pos_score - 2 * score_diff_max
    return check_num, threshold


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

    train_X, train_y, test_X, candidate_objs, train_objs = prepare_data(
        query, bm25_objs, hnsw_objs, bm25_obj_score_dict, hnsw_obj_score_dict
    )
    bm25_weights = [0.5, 0.5]
    hnsw_weights = [0.5, 0.5]
    train_X, test_X = agg_feature(train_X, test_X, bm25_weights, hnsw_weights)
    test_obj_to_feature = {}
    for obj, feature in zip(candidate_objs, test_X):
        test_obj_to_feature[obj] = list(np.round(feature, 4))

    # search the best score function
    best_w1, best_w2, train_score = search_best_params(train_X, train_y)
    print(f"best_w1: {best_w1:.2f}, best_w2: {best_w2:.2f}")
    obj_score_to_save = {}
    for obj, score in zip(train_objs, train_score):
        obj_score_to_save[obj] = score

    # determine target number of objs to check
    check_num, threshold = determine_check_num_and_threshold(train_y, train_score)
    print(f"check_num: {check_num}, score_threshold: {threshold:.4f}")

    test_score = (
        best_w1 * test_X[:, 0]
        + best_w2 * test_X[:, 1]
        + (1 - best_w1 - best_w2) * np.maximum(test_X[:, 0], test_X[:, 1])
    )
    obj_score = {}
    for obj, score in zip(candidate_objs, test_score):
        obj_score[obj] = score
        obj_score_to_save[obj] = score
    sorted_obj_score = sorted(obj_score.items(), key=lambda x: x[1], reverse=True)
    # # save the obj_score to a json file
    # with open(f"debug/chemical/{query.org_query}.json", "w") as f:
    #     json.dump(obj_score_to_save, f, indent=4)
    obj_to_check = []
    for i, (obj, score) in enumerate(sorted_obj_score):
        if (i < MIN_CHECK_NUM or score > threshold) and i < min(
            MAX_CHECK_NUM, args.budget - len(query.obj_scores)
        ):
            obj_to_check.append(obj)
        else:
            break
    for obj in obj_to_check[:5] + obj_to_check[-5:]:
        print(
            f"obj: {obj}, score: {obj_score[obj]:.2f}, feature: {test_obj_to_feature[obj]}"
        )
    return obj_to_check, len(obj_to_check) == MAX_CHECK_NUM
