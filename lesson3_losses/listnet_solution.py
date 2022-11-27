import math

import numpy as np
import torch
from catboost.datasets import msrank_10k
from sklearn.preprocessing import StandardScaler
from torch import nn

from typing import List


class ListNet(torch.nn.Module):
    def __init__(self, num_input_features: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        # укажите архитектуру простой модели здесь
        self.model = nn.Sequential(
            nn.Linear(num_input_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, input_1: torch.Tensor) -> torch.Tensor:
        logits = self.model(input_1)
        return logits


class Solution:
    def __init__(self, n_epochs: int = 5, listnet_hidden_dim: int = 30,
                 lr: float = 0.001, ndcg_top_k: int = 10):
        self._prepare_data()
        self.num_input_features = self.X_train.shape[1]
        self.ndcg_top_k = ndcg_top_k
        self.n_epochs = n_epochs

        self.model = self._create_model(
            self.num_input_features, listnet_hidden_dim
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

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
        self.ys_train = torch.from_numpy(y_train).type(torch.FloatTensor)
        self.X_test = torch.from_numpy(X_test).type(torch.FloatTensor)
        self.ys_test = torch.from_numpy(y_test).type(torch.FloatTensor)

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

    def _create_model(self, listnet_num_input_features: int,
                      listnet_hidden_dim: int) -> torch.nn.Module:
        torch.manual_seed(0)
        # допишите ваш код здесь
        net = ListNet(
            listnet_num_input_features,
            listnet_hidden_dim
        )
        return net

    def fit(self) -> List[float]:
        # допишите ваш код здесь
        val_ndcgs = []
        for _ in range(self.n_epochs):
            self._train_one_epoch()
            val_ndcgs.append(self._eval_test_set())
        return val_ndcgs

    def _calc_loss(self, batch_ys: torch.FloatTensor,
                   batch_pred: torch.FloatTensor) -> torch.FloatTensor:
        # допишите ваш код здесь
        P_y_i = torch.softmax(batch_ys, dim=0)
        P_z_i = torch.softmax(batch_pred, dim=0)
        return -torch.sum(P_y_i * torch.log(P_z_i/P_y_i))

    def _train_one_epoch(self) -> None:
        self.model.train()
        # допишите ваш код здесь
        unique_queries = np.unique(self.query_ids_train)
        np.random.shuffle(unique_queries)

        for query_id in unique_queries:
            group_X = self.X_train[self.query_ids_train == query_id]
            group_y = self.ys_train[self.query_ids_train == query_id]

            self.optimizer.zero_grad()
            preds = self.model(group_X).reshape(-1,)
            loss = self._calc_loss(group_y.reshape(-1,), preds)
            loss.backward()
            self.optimizer.step()

    def _eval_test_set(self) -> float:
        with torch.no_grad():
            self.model.eval()
            unique_queries = np.unique(self.query_ids_test)
            ndcgs = []
            # допишите ваш код здесь
            for query_id in unique_queries:
                batch_X = self.X_test[self.query_ids_test == query_id]
                batch_y = self.ys_test[self.query_ids_test == query_id]
                y_pred = self.model(batch_X)
                group_dcg = self._ndcg_k(batch_y, y_pred,
                                         self.ndcg_top_k).item()
                ndcgs.append(group_dcg)
            return np.mean(ndcgs)

    def _dcg(self, ys_true: torch.Tensor, ys_pred: torch.Tensor,
             k: int) -> float:
        ys_pred, indices = torch.sort(ys_pred, dim=0, descending=True)
        ys_true = ys_true[indices]
        sum_dcg = i = 0
        k = min(len(ys_true), k)
        while i < k:
            sum_dcg += (2 ** ys_true[i] - 1) / math.log2(i + 2)
            i += 1
        return sum_dcg

    def _ndcg_k(self, ys_true: torch.Tensor, ys_pred: torch.Tensor,
                ndcg_top_k: int) -> float:
        case_dcg = self._dcg(ys_true, ys_pred, ndcg_top_k)
        ideal_dcg = self._dcg(ys_true, ys_true, ndcg_top_k)
        return case_dcg / ideal_dcg
