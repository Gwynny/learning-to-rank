import numpy as np
from typing import Callable, Tuple, Dict, List


def distance(query: np.ndarray, documents: np.ndarray) -> np.ndarray:
    return np.linalg.norm(query - documents, axis=1).reshape(-1, 1)


def create_sw_graph(
        data: np.ndarray,
        num_candidates_for_choice_long: int = 10,
        num_edges_long: int = 5,
        num_candidates_for_choice_short: int = 10,
        num_edges_short: int = 5,
        use_sampling: bool = False,
        sampling_share: float = 0.05,
        dist_f: Callable = distance
) -> Dict[int, List[int]]:
    """
    creates small world graph with closest and furthest points
    :param data:
    :param num_candidates_for_choice_long: num furthest points from cur point
    :param num_edges_long: num sample from previous candidates
    :param num_candidates_for_choice_short: num closest points to cur point
    :param num_edges_short:
    :param use_sampling: if we have too much data we can use sampling
    :param sampling_share: proportion of data to sample
    :param dist_f: function for distance (euclidean)
    :return: sw graph
    """
    # my code here
    graph = {}
    num_points = data.shape[0]
    for i, point in enumerate(data):
        if not use_sampling:
            all_dists = dist_f(point, data)
            argsorted = np.argsort(all_dists.reshape(1, -1))[0][1:]
        else:
            sample_size = int(num_points * sampling_share)
            choiced = np.random.choice(
                list(range(num_points)), size=sample_size, replace=False)
            part_dists = dist_f(point, data[choiced, :])
            argsorted = choiced[np.argsort(part_dists.reshape(1, -1))[0][1:]]
        candidates_for_point = []

        further_points = argsorted[-num_candidates_for_choice_long:]
        further_points = np.random.choice(further_points, size=num_edges_long,
                                          replace=False)
        candidates_for_point.extend(list(further_points))

        closer_points = argsorted[:num_candidates_for_choice_short]
        closer_points = np.random.choice(closer_points, size=num_edges_short,
                                         replace=False)
        candidates_for_point.extend(list(closer_points))
        graph[i] = candidates_for_point
    return graph


def calc_dist_and_upd(all_visited_points: dict,
                      query_point: np.ndarray,
                      all_documents: np.ndarray,
                      point_idx: int,
                      dist_f: Callable
                      ) -> Tuple[float, bool]:
    if point_idx in all_visited_points:
        return all_visited_points[point_idx], True
    cur_dist = \
        dist_f(query_point, all_documents[point_idx, :].reshape(1, -1))[0][0]
    all_visited_points[point_idx] = cur_dist
    return cur_dist, False


def nsw(query_point: np.ndarray,
        all_documents: np.ndarray,
        graph_edges: Dict,
        search_k: int = 10, num_start_points: int = 5,
        dist_f: Callable = distance) -> np.ndarray:
    """
    do navigable search with help of small world graph
    :param query_point: input for which we want to find closest documents
    :param all_documents: all available data
    :param graph_edges: small world graph
    :param search_k: num of output documents
    :param num_start_points: 5 init point to start a search through graph
    :param dist_f: eucledian distance
    :return: approximate top k closest docs
    """
    all_visited_points = {}
    num_started_points = 0
    while (num_started_points < num_start_points) or \
            (len(all_visited_points) < search_k):
        cur_point_idx = np.random.randint(0, all_documents.shape[0] - 1)
        cur_dist, is_visited = calc_dist_and_upd(
            all_visited_points, query_point, all_documents,
            cur_point_idx, dist_f)
        if is_visited:
            continue

        while True:
            min_dist = cur_dist
            choiced_cand = cur_point_idx

            cands_idxs = graph_edges[cur_point_idx]
            visited_before_cands = {cur_point_idx}
            for cand_idx in cands_idxs:
                tmp_d, verdict = calc_dist_and_upd(
                    all_visited_points, query_point, all_documents,
                    cand_idx, dist_f)
                if tmp_d < min_dist:
                    min_dist = tmp_d
                    choiced_cand = cand_idx
                if is_visited:
                    visited_before_cands.add(cand_idx)

            if choiced_cand in visited_before_cands:
                break
            cur_dist = min_dist
            cur_point_idx = choiced_cand

        num_started_points += 1

    best_idxs = np.argsort(list(all_visited_points.values()))[:search_k]
    final_idx = np.array(list(all_visited_points.keys()))[best_idxs]
    return final_idx
