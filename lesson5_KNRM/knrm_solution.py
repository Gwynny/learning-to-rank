import itertools
import math
import nltk
import string
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from collections import Counter
from typing import Dict, List, Tuple, Union, Callable

nltk.download('punkt')

# Замените пути до директорий и файлов! Можете использовать для локальной
# отладки.
# При проверке на сервере пути будут изменены
glue_qqp_dir = '/data/QQP/'
glove_path = '/data/glove.6B.50d.txt'


class GaussianKernel(torch.nn.Module):
    def __init__(self, mu: float = 1., sigma: float = 1.):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x):
        numerator = -torch.pow((x - self.mu), 2)
        denominator = 2 * self.sigma ** 2
        return torch.exp(numerator / denominator)


class KNRM(torch.nn.Module):
    def __init__(self, embedding_matrix: np.ndarray,
                 freeze_embeddings: bool,
                 kernel_num: int = 21,
                 sigma: float = 0.1,
                 exact_sigma: float = 0.001,
                 out_layers: List[int] = [10, 5]):
        super().__init__()
        self.embeddings = torch.nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix),
            freeze=freeze_embeddings,
            padding_idx=0)

        self.kernel_num = kernel_num
        self.sigma = sigma
        self.exact_sigma = exact_sigma
        self.out_layers = out_layers

        self.kernels = self._get_kernels_layers()

        self.mlp = self._get_mlp()

        self.out_activation = torch.nn.Sigmoid()

    def _get_kernels_layers(self) -> torch.nn.ModuleList:
        kernels = torch.nn.ModuleList()
        # my code here
        shrink_len = 1.0 / self.kernel_num
        left, right = -1.0 + shrink_len, 1.0 - shrink_len
        mus = np.append(np.linspace(left, right, self.kernel_num - 1), 1.0)
        sigmas = np.array(
            (self.kernel_num - 1) * [self.sigma] + [self.exact_sigma])

        for mu, sigma in zip(mus, sigmas):
            kernels.append(GaussianKernel(mu=mu, sigma=sigma))
        return kernels

    def _get_mlp(self) -> torch.nn.Sequential:
        # my code here
        if len(self.out_layers) == 0:
            return torch.nn.Sequential(torch.nn.Linear(self.kernel_num, 1))
        layers = [torch.nn.Linear(self.kernel_num, self.out_layers[0]),
                  torch.nn.ReLU()]
        for i in range(1, len(self.out_layers)):
            layers.append(
                torch.nn.Linear(self.out_layers[i - 1], self.out_layers[i]))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(self.out_layers[-1], 1))
        return torch.nn.Sequential(*layers)

    def forward(self, input_1: Dict[str, torch.Tensor],
                input_2: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        logits_1 = self.predict(input_1)
        logits_2 = self.predict(input_2)

        logits_diff = logits_1 - logits_2

        out = self.out_activation(logits_diff)
        return out

    def _get_matching_matrix(self, query: torch.Tensor,
                             doc: torch.Tensor) -> torch.FloatTensor:
        # my code here
        # https://stackoverflow.com/questions/50411191/
        # how-to-compute-the-cosine-similarity-in-pytorch-for-all-rows-in-a
        # -matrix-with-re
        eps = 1e-8
        query_m, doc_m = self.embeddings(query), self.embeddings(doc)
        query_norm = query_m.norm(dim=2)[:, :, None]
        doc_norm = doc_m.norm(dim=2)[:, :, None]
        query_normalised = query_m / torch.clamp(query_norm, min=eps)
        doc_normalised = doc_m / torch.clamp(doc_norm, min=eps)
        similarity_m = torch.bmm(query_normalised, doc_normalised.transpose(1, 2))
        return similarity_m

    def _apply_kernels(self,
                       matching_matrix: torch.FloatTensor) -> \
            torch.FloatTensor:
        KM = []
        for kernel in self.kernels:
            # shape = [B]
            K = torch.log1p(kernel(matching_matrix).sum(dim=-1)).sum(dim=-1)
            KM.append(K)

        # shape = [B, K]
        kernels_out = torch.stack(KM, dim=1)
        return kernels_out

    def predict(self, inputs: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        # shape = [Batch, Left], [Batch, Right]
        query, doc = inputs['query'], inputs['document']

        # shape = [Batch, Left, Right]
        matching_matrix = self._get_matching_matrix(query, doc)
        # shape = [Batch, Kernels]
        kernels_out = self._apply_kernels(matching_matrix)
        # shape = [Batch]
        out = self.mlp(kernels_out)
        return out


class RankingDataset(torch.utils.data.Dataset):
    def __init__(self, index_pairs_or_triplets: List[List[Union[str, float]]],
                 idx_to_text_mapping: Dict[str, str], vocab: Dict[str, int],
                 oov_val: int,
                 preproc_func: Callable, max_len: int = 30):
        self.index_pairs_or_triplets = index_pairs_or_triplets
        self.idx_to_text_mapping = idx_to_text_mapping
        self.vocab = vocab
        self.oov_val = oov_val
        self.preproc_func = preproc_func
        self.max_len = max_len

    def __len__(self):
        return len(self.index_pairs_or_triplets)

    def _tokenized_text_to_index(self, tokenized_text: List[str]) -> List[int]:
        # my code here
        token_idxs = []
        text = tokenized_text[:self.max_len]
        for token in text:
            token_idxs.append(self.vocab.get(token, self.oov_val))
        return token_idxs

    def _convert_text_idx_to_token_idxs(self, idx: int) -> List[int]:
        # my code here
        text = self.idx_to_text_mapping[idx]
        tokenized_text = self.preproc_func(text)
        token_idxs = self._tokenized_text_to_index(tokenized_text)
        return token_idxs

    def __getitem__(self, idx: int):
        pass

    def __getitem__(self, idx: int):
        pass


class TrainTripletsDataset(RankingDataset):
    def __getitem__(self, idx):
        # my code here
        triplets = self.index_pairs_or_triplets[idx]
        query_tokens = self._convert_text_idx_to_token_idxs(str(triplets[0]))
        left_doc_tokens = self._convert_text_idx_to_token_idxs(str(triplets[1]))
        right_doc_tokens = self._convert_text_idx_to_token_idxs(str(triplets[2]))
        label = triplets[3]

        left_query_doc = {'query': query_tokens, 'document': left_doc_tokens}
        right_query_doc = {'query': query_tokens, 'document': right_doc_tokens}
        return left_query_doc, right_query_doc, label


class ValPairsDataset(RankingDataset):
    def __getitem__(self, idx):
        # my code here
        pairs = self.index_pairs_or_triplets[idx]
        query_tokens = self._convert_text_idx_to_token_idxs(str(pairs[0]))
        doc_tokens = self._convert_text_idx_to_token_idxs(str(pairs[1]))
        label = pairs[2]

        query_doc = {'query': query_tokens, 'document': doc_tokens}
        return query_doc, label


def collate_fn(
        batch_objs: List[Union[Dict[str, torch.Tensor], torch.FloatTensor]]):
    max_len_q1 = -1
    max_len_d1 = -1
    max_len_q2 = -1
    max_len_d2 = -1

    is_triplets = False
    for elem in batch_objs:
        if len(elem) == 3:
            left_elem, right_elem, label = elem
            is_triplets = True
        else:
            left_elem, label = elem

        max_len_q1 = max(len(left_elem['query']), max_len_q1)
        max_len_d1 = max(len(left_elem['document']), max_len_d1)
        if len(elem) == 3:
            max_len_q2 = max(len(right_elem['query']), max_len_q2)
            max_len_d2 = max(len(right_elem['document']), max_len_d2)

    q1s = []
    d1s = []
    q2s = []
    d2s = []
    labels = []

    for elem in batch_objs:
        if is_triplets:
            left_elem, right_elem, label = elem
        else:
            left_elem, label = elem

        pad_len1 = max_len_q1 - len(left_elem['query'])
        pad_len2 = max_len_d1 - len(left_elem['document'])
        if is_triplets:
            pad_len3 = max_len_q2 - len(right_elem['query'])
            pad_len4 = max_len_d2 - len(right_elem['document'])

        q1s.append(left_elem['query'] + [0] * pad_len1)
        d1s.append(left_elem['document'] + [0] * pad_len2)
        if is_triplets:
            q2s.append(right_elem['query'] + [0] * pad_len3)
            d2s.append(right_elem['document'] + [0] * pad_len4)
        labels.append([label])
    q1s = torch.LongTensor(q1s)
    d1s = torch.LongTensor(d1s)
    if is_triplets:
        q2s = torch.LongTensor(q2s)
        d2s = torch.LongTensor(d2s)
    labels = torch.FloatTensor(labels)

    ret_left = {'query': q1s, 'document': d1s}
    if is_triplets:
        ret_right = {'query': q2s, 'document': d2s}
        return ret_left, ret_right, labels
    else:
        return ret_left, labels


class Solution:
    def __init__(self, glue_qqp_dir: str,
                 glove_vectors_path: str,
                 min_token_occurancies: int = 1,
                 random_seed: int = 0,
                 emb_rand_uni_bound: float = 0.2,
                 freeze_knrm_embeddings: bool = True,
                 knrm_kernel_num: int = 21,
                 knrm_out_mlp: List[int] = [],
                 dataloader_bs: int = 1024,
                 train_lr: float = 0.001,
                 change_train_loader_ep: int = 10
                 ):
        self.glue_qqp_dir = glue_qqp_dir
        self.glove_vectors_path = glove_vectors_path
        self.glue_train_df = self.get_glue_df('train')
        self.glue_dev_df = self.get_glue_df('dev')
        self.dev_pairs_for_ndcg = self.create_val_pairs(self.glue_dev_df)
        self.min_token_occurancies = min_token_occurancies
        self.all_tokens = self.get_all_tokens(
            [self.glue_train_df, self.glue_dev_df], self.min_token_occurancies)

        self.random_seed = random_seed
        self.emb_rand_uni_bound = emb_rand_uni_bound
        self.freeze_knrm_embeddings = freeze_knrm_embeddings
        self.knrm_kernel_num = knrm_kernel_num
        self.knrm_out_mlp = knrm_out_mlp
        self.dataloader_bs = dataloader_bs
        self.train_lr = train_lr
        self.change_train_loader_ep = change_train_loader_ep

        self.model, self.vocab, self.unk_words = self.build_knrm_model()
        self.idx_to_text_mapping_train = self.get_idx_to_text_mapping(
            self.glue_train_df)
        self.idx_to_text_mapping_dev = self.get_idx_to_text_mapping(
            self.glue_dev_df)

        self.val_dataset = ValPairsDataset(self.dev_pairs_for_ndcg,
                                           self.idx_to_text_mapping_dev,
                                           vocab=self.vocab,
                                           oov_val=self.vocab['OOV'],
                                           preproc_func=self.simple_preproc)
        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.dataloader_bs, num_workers=0,
            collate_fn=collate_fn, shuffle=False)

    def get_glue_df(self, partition_type: str) -> pd.DataFrame:
        assert partition_type in ['dev', 'train']
        glue_df = pd.read_csv(
            self.glue_qqp_dir + f'/{partition_type}.tsv', sep='\t',
            error_bad_lines=False, dtype=object)
        glue_df = glue_df.dropna(axis=0, how='any').reset_index(drop=True)
        glue_df_fin = pd.DataFrame({
            'id_left': glue_df['qid1'],
            'id_right': glue_df['qid2'],
            'text_left': glue_df['question1'],
            'text_right': glue_df['question2'],
            'label': glue_df['is_duplicate'].astype(int)
        })
        return glue_df_fin

    def handle_punctuation(self, inp_str: str) -> str:
        # my code below
        translator = str.maketrans(string.punctuation,
                                   ' ' * len(string.punctuation))
        new_str = inp_str.translate(translator)
        return new_str

    def simple_preproc(self, inp_str: str) -> List[str]:
        # my code below
        no_punctuation_str = self.handle_punctuation(inp_str)
        lowered_str = no_punctuation_str.lower()
        splitted_doc = nltk.word_tokenize(lowered_str)
        return splitted_doc

    def _filter_rare_words(self, vocab: Dict[str, int],
                           min_occurancies: int) -> Dict[str, int]:
        # my code below
        filtered_vocab = {x: count for x, count in vocab.items() if
                          count >= min_occurancies}
        return filtered_vocab

    def get_all_tokens(self, list_of_df: List[pd.DataFrame],
                       min_occurancies: int) -> List[str]:
        # my code below
        preped_series = []
        for df in list_of_df:
            preped_question1 = df['text_left'].apply(self.simple_preproc)
            preped_question2 = df['text_right'].apply(self.simple_preproc)
            preped_series.append(preped_question1)
            preped_series.append(preped_question2)

        concat_series = pd.concat(preped_series)
        one_list_of_tokens = list(
            itertools.chain.from_iterable(concat_series.to_list())
        )
        vocab = dict(Counter(one_list_of_tokens))
        vocab = self._filter_rare_words(vocab, min_occurancies)
        tokens = [key for key in vocab.keys()]
        return tokens

    def _read_glove_embeddings(self, file_path: str) -> Dict[str, List[str]]:
        # my code below
        with open(file_path, encoding='utf-8') as file:
            glove_dict = {}
            for line in file:
                splitted_line = line.split()
                word, embedding = splitted_line[0], splitted_line[1:]
                glove_dict[word] = embedding
        return glove_dict

    def create_glove_emb_from_file(self, file_path: str,
                                   inner_keys: List[str],
                                   random_seed: int,
                                   rand_uni_bound: float
                                   ) -> Tuple[np.ndarray,
                                              Dict[str, int],
                                              List[str]]:
        # my code below
        np.random.seed(random_seed)
        glove_dict = self._read_glove_embeddings(file_path)
        emb_dim = len(glove_dict['the'])

        emb_matrix = []
        pad_vec = np.random.uniform(low=-rand_uni_bound,
                                    high=rand_uni_bound,
                                    size=emb_dim)
        oov_vec = np.random.uniform(low=-rand_uni_bound,
                                    high=rand_uni_bound,
                                    size=emb_dim)
        emb_matrix.append(pad_vec)
        emb_matrix.append(oov_vec)

        vocab = {}
        unk_words = []
        vocab['PAD'], vocab['OOV'] = 0, 1
        for ind, token in enumerate(inner_keys, 2):
            if token in glove_dict.keys():
                emb_matrix.append(glove_dict[token])
                vocab[token] = ind
            else:
                random_emb = np.random.uniform(low=-rand_uni_bound,
                                               high=rand_uni_bound,
                                               size=emb_dim)
                emb_matrix.append(random_emb)
                unk_words.append(token)
                vocab[token] = ind
        emb_matrix = np.array(emb_matrix).astype(float)
        return emb_matrix, vocab, unk_words

    def build_knrm_model(self) -> Tuple[
        torch.nn.Module, Dict[str, int], List[str]]:
        emb_matrix, vocab, unk_words = \
            self.create_glove_emb_from_file(self.glove_vectors_path,
                                            self.all_tokens,
                                            self.random_seed,
                                            self.emb_rand_uni_bound)
        torch.manual_seed(self.random_seed)
        knrm = KNRM(emb_matrix,
                    freeze_embeddings=self.freeze_knrm_embeddings,
                    out_layers=self.knrm_out_mlp,
                    kernel_num=self.knrm_kernel_num)
        return knrm, vocab, unk_words

    def sample_data_for_train_iter(inp_df: pd.DataFrame, seed: int
                                   ) -> List[List[Union[str, float]]]:
        # допишите ваш код здесь
        inp_df_select = train_df[['id_left', 'id_right', 'label']]
        inf_df_group_sizes = inp_df_select.groupby('id_left').size()
        glue_dev_leftids_to_use = list(
            inf_df_group_sizes[inf_df_group_sizes >= 3].index)
        glue_dev_leftids_to_use = np.random.choice(
            list(glue_dev_leftids_to_use), size=3000, replace=False)
        groups = inp_df_select[inp_df_select.id_left.isin(
            glue_dev_leftids_to_use)].groupby('id_left')

        all_ids = set(train_df['id_left']).union(set(train_df['id_right']))

        out_triplets = []

        np.random.seed(seed)
        negative_example = np.random.choice(list(all_ids), size=1).item()
        for id_left, group in groups:
            right_ids = np.array(group['id_right'].to_list())
            np.random.shuffle(right_ids)
            all_groups_ids = set([id_left]).union(set(right_ids))
            candidates = list(combinations(right_ids, 2))
            candidates_inds = np.random.choice(list(range(len(candidates))),
                                               size=3, replace=False)

            for ind in candidates_inds:
                candidate_left, candidate_right = candidates[ind][0], \
                                                  candidates[ind][1]
                left_label = group[group['id_right'] == candidate_left][
                    'label'].item()
                right_label = group[group['id_right'] == candidate_right][
                    'label'].item()
                label_diff = left_label - right_label
                if label_diff > 0:
                    out_triplets.append(
                        [id_left, candidate_left, candidate_right, 1])
                else:
                    out_triplets.append(
                        [id_left, candidate_left, candidate_right, 0])

            # negative_example = np.random.choice(list(all_ids), size=1).item()
            out_triplets.append(
                [id_left, candidate_right, negative_example, 0])
            negative_example = id_left

        out_triplets = np.array(out_triplets)
        train_inds = np.random.choice(list(range(len(out_triplets))),
                                      size=10000, replace=False)
        return out_triplets[train_inds]

    def create_val_pairs(self,
                         inp_df: pd.DataFrame,
                         fill_top_to: int = 15,
                         min_group_size: int = 2,
                         seed: int = 0) -> List[List[Union[str, float]]]:
        inp_df_select = inp_df[['id_left', 'id_right', 'label']]
        inf_df_group_sizes = inp_df_select.groupby('id_left').size()
        glue_dev_leftids_to_use = list(
            inf_df_group_sizes[inf_df_group_sizes >= min_group_size].index)
        groups = inp_df_select[inp_df_select.id_left.isin(
            glue_dev_leftids_to_use)].groupby('id_left')

        all_ids = set(inp_df['id_left']).union(set(inp_df['id_right']))

        out_pairs = []

        np.random.seed(seed)

        for id_left, group in groups:
            ones_ids = group[group.label > 0].id_right.values
            zeroes_ids = group[group.label == 0].id_right.values
            sum_len = len(ones_ids) + len(zeroes_ids)
            num_pad_items = max(0, fill_top_to - sum_len)
            if num_pad_items > 0:
                cur_chosen = set(ones_ids).union(
                    set(zeroes_ids)).union({id_left})
                pad_sample = np.random.choice(
                    list(all_ids - cur_chosen), num_pad_items,
                    replace=False).tolist()
            else:
                pad_sample = []
            for i in ones_ids:
                out_pairs.append([id_left, i, 2])
            for i in zeroes_ids:
                out_pairs.append([id_left, i, 1])
            for i in pad_sample:
                out_pairs.append([id_left, i, 0])
        return out_pairs

    def get_idx_to_text_mapping(self, inp_df: pd.DataFrame) -> Dict[str, str]:
        left_dict = (
            inp_df[
                ['id_left', 'text_left']
            ].drop_duplicates()
                .set_index('id_left')
            ['text_left'].to_dict()
        )
        right_dict = (
            inp_df[
                ['id_right', 'text_right']
            ].drop_duplicates()
                .set_index('id_right')
            ['text_right'].to_dict()
        )
        left_dict.update(right_dict)
        return left_dict

    def _dcg(self, ys_true: np.array, ys_pred: np.array, k: int) -> float:
        indices = np.argsort(-ys_pred)
        ys_true = ys_true[indices[:k]]

        sum_dcg = 0
        for i, y_true in enumerate(ys_true, 1):
            sum_dcg += (2 ** y_true - 1) / math.log2(i + 1)
        return sum_dcg

    def ndcg_k(self, ys_true: np.array, ys_pred: np.array,
               ndcg_top_k: int = 10) -> float:
        ideal_dcg = self._dcg(ys_true, ys_true, ndcg_top_k)
        case_dcg = self._dcg(ys_true, ys_pred, ndcg_top_k)
        return float(case_dcg / ideal_dcg)

    def valid(self, model: torch.nn.Module,
              val_dataloader: torch.utils.data.DataLoader) -> float:
        labels_and_groups = val_dataloader.dataset.index_pairs_or_triplets
        labels_and_groups = pd.DataFrame(labels_and_groups,
                                         columns=['left_id', 'right_id',
                                                  'rel'])

        all_preds = []
        for batch in (val_dataloader):
            inp_1, y = batch
            preds = model.predict(inp_1)
            preds_np = preds.detach().numpy()
            all_preds.append(preds_np)
        all_preds = np.concatenate(all_preds, axis=0)
        labels_and_groups['preds'] = all_preds

        ndcgs = []
        for cur_id in labels_and_groups.left_id.unique():
            cur_df = labels_and_groups[labels_and_groups.left_id == cur_id]
            ndcg = self.ndcg_k(cur_df.rel.values.reshape(-1),
                               cur_df.preds.values.reshape(-1))
            if np.isnan(ndcg):
                ndcgs.append(0)
            else:
                ndcgs.append(ndcg)
        return np.mean(ndcgs)

    def train(self, n_epochs: int):
        opt = torch.optim.SGD(self.model.parameters(), lr=self.train_lr)
        criterion = torch.nn.BCELoss()
        # допишите ваш код здесь
        triplets = sample_data_for_train_iter(self.glue_train_df, 0)
        train_dataset = TrainTripletsDataset(triplets,
                                             self.idx_to_text_mapping_train,
                                             vocab=self.vocab,
                                             oov_val=self.vocab['OOV'],
                                             preproc_func=self.simple_preproc)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.dataloader_bs, num_workers=0,
            collate_fn=collate_fn, shuffle=True)

        for i in range(n_epochs):
            self.model.train(True)
            for j, data in enumerate(train_dataloader):
                # Every data instance is an input + label pair
                query_left_docs, query_right_docs, labels = data
                opt.zero_grad()
                outputs = self.model(query_left_docs, query_right_docs)
                loss = criterion(outputs, labels)
                loss.backward()
                opt.step()

            self.model.train(False)
            val_ndcg = self.valid(self.model, self.val_dataloader)
            print(f'Epoch: {i}, validation ndcg {val_ndcg}')
