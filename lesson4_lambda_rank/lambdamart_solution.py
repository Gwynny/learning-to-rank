import math
import pickle
import random
from typing import List, Tuple

import numpy as np
import torch
from catboost.datasets import msrank_10k
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from tqdm.auto import tqdm


class Solution:
    def __init__(self, n_estimators: int = 100, lr: float = 0.5,
                 ndcg_top_k: int = 10,
                 subsample: float = 0.6, colsample_bytree: float = 0.9,
                 max_depth: int = 5, min_samples_leaf: int = 8):
        self._prepare_data()

        self.ndcg_top_k = ndcg_top_k
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

        # допишите ваш код здесь
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.trees = []

    def _get_data(self) -> List[np.ndarray]:
        train_df, test_df = msrank_10k()

        X_train = train_df.drop([0, 1], axis=1).values
        y_train = train_df[0].values
        query_ids_train = train_df[1].values.astype(int)

        X_test = test_df.drop([0, 1], axis=1).values
        y_test = test_df[0].values
        query_ids_test = test_df[1].values.astype(int)

        return [X_train, y_train, query_ids_train, X_test, y_test,
                query_ids_test]

    def _prepare_data(self) -> None:
        (X_train, y_train, self.query_ids_train,
         X_test, y_test, self.query_ids_test) = self._get_data()
        # допишите ваш код здесь
        X_train = self._scale_features_in_query_groups(
            X_train, self.query_ids_train
        )
        X_test = self._scale_features_in_query_groups(
            X_test, self.query_ids_test
        )

        self.X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
        self.ys_train = torch.from_numpy(y_train).type(
            torch.FloatTensor).reshape(-1, 1)
        self.X_test = torch.from_numpy(X_test).type(torch.FloatTensor)
        self.ys_test = torch.from_numpy(y_test).type(
            torch.FloatTensor).reshape(-1, 1)

    def _scale_features_in_query_groups(self, inp_feat_array: np.ndarray,
                                        inp_query_ids: np.ndarray) -> \
            np.ndarray:
        # допишите ваш код здесь
        for query_id in np.unique(inp_query_ids):
            mask = inp_query_ids == query_id
            scaler = StandardScaler()
            scaled_part = scaler.fit_transform(inp_feat_array[mask])
            inp_feat_array[mask] = scaled_part
        return inp_feat_array

    def _train_one_tree(self, cur_tree_idx: int,
                        train_preds: torch.FloatTensor
                        ) -> Tuple[DecisionTreeRegressor, np.ndarray]:
        # допишите ваш код здесь
        np.random.seed(cur_tree_idx)
        unique_queries = np.unique(self.query_ids_train)
        num_objects = self.X_train.shape[0]
        lambdas = np.zeros((num_objects, 1))
        for i, query_id in enumerate(unique_queries):
            mask = self.query_ids_train == query_id
            group_X = self.X_train[mask]
            group_y = self.ys_train[mask]
            group_preds = train_preds[mask]
            group_lambdas = self._compute_lambdas(group_y, group_preds)
            lambdas[mask] = group_lambdas

        dt = DecisionTreeRegressor(max_depth=self.max_depth,
                                   min_samples_leaf=self.min_samples_leaf,
                                   random_state=cur_tree_idx)
        num_features = self.X_train.shape[1]
        rows = np.random.choice(np.arange(num_objects),
                                size=int(self.subsample * num_objects),
                                replace=False)
        cols = np.random.choice(np.arange(num_features),
                                size=int(self.colsample_bytree * num_features),
                                replace=False)

        sample_X_train = self.X_train[rows, :][:, cols]
        lambdas = lambdas[rows]
        dt.fit(sample_X_train, lambdas)
        return (dt, cols)

    def fit(self):
        np.random.seed(0)
        # допишите ваш код здесь
        num_objects = self.X_train.shape[0]
        best_ndcg = max_ndcg = 0
        prev_preds = torch.from_numpy(np.zeros((num_objects, 1))).type(
            torch.FloatTensor)
        valid_preds = torch.from_numpy(
            np.zeros((self.X_test.shape[0], 1))).type(torch.FloatTensor)

        for idx in range(1, self.n_estimators + 1):
            dt, train_cols = self._train_one_tree(idx, prev_preds)
            self.trees.append((dt, train_cols))
            prev_preds -= self.lr * dt.predict(
                self.X_train[:, train_cols]).reshape(-1, 1)
            valid_preds -= self.lr * dt.predict(
                self.X_test[:, train_cols]).reshape(-1, 1)
            ndcg = self._calc_data_ndcg(idx, self.ys_test, valid_preds)

            if ndcg > max_ndcg:
                best_ndcg = idx
            print(ndcg)

        self.trees = self.trees[:best_ndcg]

    def predict(self, data: torch.FloatTensor) -> torch.FloatTensor:
        preds = torch.from_numpy(np.zeros((self.X_test.shape[0], 1))).type(
            torch.FloatTensor)
        for dt, cols in self.trees:
            preds -= self.lr * dt.predict(data[:, cols])
        return preds

    def _compute_lambdas(self, y_true: torch.FloatTensor,
                         y_pred: torch.FloatTensor) -> torch.FloatTensor:
        def compute_ideal_dcg(ys_true: torch.Tensor) -> float:
            ys_true, _ = torch.sort(ys_true, dim=0, descending=True)

            sum_dcg = 0
            for i, y_true in enumerate(ys_true, 1):
                sum_dcg += (2 ** y_true - 1) / math.log2(i + 1)
            return sum_dcg

        def compute_labels_in_batch(y_true):
            rel_diff = y_true - y_true.t()
            pos_pairs = (rel_diff > 0).type(torch.float32)
            neg_pairs = (rel_diff < 0).type(torch.float32)
            Sij = pos_pairs - neg_pairs
            return Sij

        _, rank_order = torch.sort(y_true, descending=True, axis=0)
        rank_order += 1

        pos_pairs_score_diff = 1.0 + torch.exp((y_pred - y_pred.t()))

        Sij = compute_labels_in_batch(y_true)

        gain_diff = torch.pow(2.0, y_true) - torch.pow(2.0, y_true.t())
        decay_diff = (1.0 / torch.log2(rank_order + 1.0)) - (
                    1.0 / torch.log2(rank_order.t() + 1.0))
        ideal_dcg = compute_ideal_dcg(y_true)
        N = 1 / (ideal_dcg + 1)
        delta_ndcg = torch.abs(N * gain_diff * decay_diff)

        lambda_update = (0.5 * (
                    1 - Sij) - 1 / pos_pairs_score_diff) * delta_ndcg
        lambda_update = torch.sum(lambda_update, dim=1, keepdim=True)
        return lambda_update

    def _dcg(self, ys_true: torch.Tensor, ys_pred: torch.Tensor,
             k: int) -> float:
        ys_pred, indices = torch.sort(ys_pred, dim=0, descending=True)
        ys_true = ys_true[indices[:k]]

        sum_dcg = 0
        for i, y_true in enumerate(ys_true, 1):
            sum_dcg += (2 ** y_true - 1) / math.log2(i + 1)
        return sum_dcg

    def _ndcg_k(self, ys_true: torch.Tensor, ys_pred: torch.Tensor,
                ndcg_top_k: int) -> float:
        ideal_dcg = self._dcg(ys_true, ys_true, ndcg_top_k)
        case_dcg = self._dcg(ys_true, ys_pred, ndcg_top_k)
        return case_dcg / ideal_dcg

    def _calc_data_ndcg(self, queries_list: np.ndarray,
                        true_labels: torch.FloatTensor,
                        preds: torch.FloatTensor) -> float:
        # допишите ваш код здесь
        unique_queries = np.unique(self.query_ids_test)
        ndcgs = []
        for query_id in unique_queries:
            group_y = true_labels[self.query_ids_test == query_id]
            y_pred = preds[self.query_ids_test == query_id]
            group_dcg = self._ndcg_k(group_y, y_pred, self.ndcg_top_k).item()
            if np.isnan(group_dcg):
                ndcgs.append(0)
                continue
            ndcgs.append(group_dcg)
        return np.mean(ndcgs)

    def save_model(self, path: str):
        # допишите ваш код здесь
        pass

    def load_model(self, path: str):
        # допишите ваш код здесь
        pass
