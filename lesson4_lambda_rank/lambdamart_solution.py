import math
import pickle
from typing import List, Tuple

import numpy as np
import torch
from catboost.datasets import msrank_10k
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from torch import Tensor


class Solution:
    def __init__(self, n_estimators: int = 231,
                 lr: float = 0.39127022258826616,
                 ndcg_top_k: int = 10,
                 subsample: float = 0.53005362889185,
                 colsample_bytree: float = 0.6304279014613413,
                 max_depth: int = 9,
                 min_samples_leaf: int = 25):
        self._prepare_data()
        self.num_input_features = self.X_train.shape[1]
        self.num_train_objects = self.X_train.shape[0]
        self.unique_train_groups = np.unique(self.query_ids_train)
        self.num_test_objects = self.X_test.shape[0]
        self.num_features_to_choice = int(colsample_bytree *
                                          self.num_input_features)
        self.num_groups_to_choice = int(
            subsample * len(self.unique_train_groups))

        self.ndcg_top_k = ndcg_top_k
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

        self.trees = []
        self.best_ndcg = -1

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
        lambdas = np.zeros((self.num_train_objects, 1))
        groups_to_train_on = np.random.choice(self.unique_train_groups,
                                              size=self.num_groups_to_choice,
                                              replace=False)
        for query_id in groups_to_train_on:
            mask = self.query_ids_train == query_id
            group_y = self.ys_train[mask]
            group_preds = train_preds[mask]
            group_lambdas = self._compute_lambdas(group_y, group_preds)
            lambdas[mask] = group_lambdas.numpy()

        dt = DecisionTreeRegressor(max_depth=self.max_depth,
                                   min_samples_leaf=self.min_samples_leaf,
                                   random_state=cur_tree_idx)

        rows = np.isin(self.query_ids_train, groups_to_train_on)
        cols = np.random.choice(np.arange(self.num_input_features),
                                size=self.num_features_to_choice,
                                replace=False)

        sample_X_train = self.X_train[rows][:, cols].numpy()
        lambdas = lambdas[rows]
        dt.fit(sample_X_train, lambdas)
        return dt, cols

    def fit(self):
        np.random.seed(0)
        # допишите ваш код здесь
        best_ndcg_ind = 0
        prev_preds = torch.zeros(
            self.num_train_objects, 1).type(torch.FloatTensor)
        valid_preds = torch.zeros(
            self.num_test_objects, 1).type(torch.FloatTensor)

        for idx in range(1, self.n_estimators + 1):
            dt, train_cols = self._train_one_tree(idx, prev_preds)
            self.trees.append((dt, train_cols))
            prev_preds -= self.lr * torch.FloatTensor(dt.predict(
                self.X_train[:, train_cols].numpy())).reshape(-1, 1)
            valid_preds -= self.lr * torch.FloatTensor(dt.predict(
                self.X_test[:, train_cols].numpy())).reshape(-1, 1)
            ndcg = self._calc_data_ndcg(self.query_ids_test, self.ys_test,
                                        valid_preds)

            if ndcg > self.best_ndcg:
                best_ndcg_ind = idx
                self.best_ndcg = ndcg

        self.trees = self.trees[:best_ndcg_ind]

    def predict(self, data: torch.FloatTensor) -> torch.FloatTensor:
        preds = torch.zeros(data.shape[0], 1).type(torch.FloatTensor)
        for dt, cols in self.trees:
            tmp_preds = dt.predict(data[:, cols].numpy())
            preds -= self.lr * torch.FloatTensor(tmp_preds).reshape(-1, 1)
        return preds

    def _compute_lambdas(self, y_true: torch.FloatTensor,
                         y_pred: torch.FloatTensor) -> Tensor:
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

        _, rank_order = torch.sort(y_true, descending=True, dim=0)
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
        return float(case_dcg / ideal_dcg)

    def _calc_data_ndcg(self, queries_list: np.ndarray,
                        true_labels: torch.FloatTensor,
                        preds: torch.FloatTensor) -> float:
        # допишите ваш код здесь
        unique_queries = np.unique(queries_list)
        ndcgs = []
        for query_id in unique_queries:
            group_y = true_labels[queries_list == query_id]
            y_pred = preds[queries_list == query_id]
            group_dcg = self._ndcg_k(group_y, y_pred, self.ndcg_top_k)
            if np.isnan(group_dcg):
                ndcgs.append(0)
                continue
            ndcgs.append(group_dcg)
        return float(np.mean(ndcgs))

    def save_model(self, path: str):
        state = {
            'trees': self.trees,
            'lr': self.lr,
            'best_ndcg': self.best_ndcg
        }
        f = open(path, 'wb')
        pickle.dump(state, f)

    def load_model(self, path: str):
        f = open(path, 'rb')
        state = pickle.load(f)
        self.trees = state['trees']
        self.lr = state['lr']
        self.best_ndcg = state['best_ndcg']
