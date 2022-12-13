from collections import OrderedDict, defaultdict
from typing import Callable, Tuple, Dict, List

import numpy as np


def distance(pointA: np.ndarray, documents: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pointA - documents, axis=1).reshape(-1, 1)


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
    for i, point in enumerate(data):
        candidates_for_point = []

        distances = dist_f(point, data)
        sorted_distances = np.argsort(distances, axis=0).reshape(-1, )

        further_points = sorted_distances[-num_candidates_for_choice_long:]
        further_points = np.random.choice(further_points, size=num_edges_long,
                                          replace=False)
        candidates_for_point.extend(list(further_points))

        closer_points = sorted_distances[1:num_candidates_for_choice_short + 1]
        closer_points = np.random.choice(closer_points, size=num_edges_short,
                                         replace=False)
        candidates_for_point.extend(list(closer_points))
        graph[i] = candidates_for_point
    return graph


def nsw(query_point: np.ndarray,
        all_documents: np.ndarray,
        graph_edges: Dict[int, List[int]],
        search_k: int = 10,
        num_start_points: int = 5,
        dist_f: Callable = distance) -> np.ndarray:
    """
    do navigable search with help of small world graph
    :param query_point: input for which we want to find closest documents
    :param all_documents: all available data
    :param graph_edges: small world graph
    :param search_k: num of output documents
    :param num_start_points: 5 init point to start a search thru graph
    :param dist_f: eucledian distance
    :return: top k closest docs
    """
    # my code below
    start_points = np.random.choice(all_documents.shape[0],
                                    size=num_start_points,
                                    replace=False)

    def search_candidates(query, graph, next_elem, visited, first_min_elem,
                          second_min_elem):
        visited.append(next_elem)
        documents = all_documents[graph[start_point]]
        distances = dist_f(query, documents)
        sorted_distances = np.argsort(distances, axis=0).reshape(-1, )
        first_closest_candidate_ind = sorted_distances[0]
        second_closest_candidate_ind = sorted_distances[1]
        first_closest_candidate_dist = distances[first_closest_candidate_ind]
        second_closest_candidate_dist = distances[second_closest_candidate_ind]

        if first_min_elem['dist'] > first_closest_candidate_dist:
            first_min_elem['ind'] = first_closest_candidate_ind
            first_min_elem['dist'] = first_closest_candidate_dist
            next_elem = first_closest_candidate_ind
        elif second_min_elem['dist'] > second_closest_candidate_dist:
            second_min_elem['ind'] = second_closest_candidate_ind
            second_min_elem['dist'] = second_closest_candidate_dist
            next_elem = second_closest_candidate_ind
        elif second_min_elem['dist'] > first_closest_candidate_dist:
            second_min_elem['ind'] = first_closest_candidate_ind
            second_min_elem['dist'] = first_closest_candidate_dist
            next_elem = first_closest_candidate_ind

        if next_elem in visited:
            print(first_min_elem)
            print(second_min_elem)
            return first_min_elem, second_min_elem
        search_candidates(query, graph, next_elem, visited, first_min_elem,
                          second_min_elem)
        # return first_min_elem, second_min_elem

    closest_candidates = []
    distances = []
    for start_point in start_points:
        first_elem = {'ind': -1, 'dist': float('inf')}
        second_elem = {'ind': -1, 'dist': float('inf')}
        visited = []
        first_min_elem, second_min_elem = search_candidates(query_point,
                                                            graph_edges,
                                                            start_point,
                                                            visited,
                                                            first_elem,
                                                            second_elem)
        closest_candidates.extend[
            [first_min_elem['ind'], second_min_elem['ind']]]
        distances.extend[[first_min_elem['dist'], second_min_elem['dist']]]

    top_closest_points = np.argsort(np.array(distances), axis=0)[:search_k]
    closest_candidates = np.array(closest_candidates)[top_closest_points]
    return all_documents[closest_candidates]

