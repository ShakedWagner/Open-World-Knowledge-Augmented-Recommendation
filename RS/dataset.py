import torch
import torch.utils.data as Data
import pickle
from utils import load_json, load_pickle
import random


class AmzDataset(Data.Dataset):
    def __init__(self, data_path, set_name='train', task='ctr', max_hist_len=10, augment=False, aug_prefix=None, user_cold_start=False, cold_start_ratio=0.0, cold_start_n_interact=1):
        self.task = task
        self.max_hist_len = max_hist_len
        self.augment = augment
        self.set = set_name
        self.data = load_pickle(data_path + f'/{task}.{set_name}')
        self.stat = load_json(data_path + '/stat.json')
        self.item_num = self.stat['item_num']
        self.attr_num = self.stat['attribute_num']
        self.attr_ft_num = self.stat['attribute_ft_num']
        self.rating_num = self.stat['rating_num']
        self.dense_dim = self.stat['dense_dim']
        if task == 'rerank':
            self.max_list_len = self.stat['rerank_list_len']
        self.length = len(self.data)
        self.sequential_data = load_json(data_path + '/sequential_data.json')
        self.item2attribution = load_json(data_path + '/item2attributes.json')
        datamaps = load_json(data_path + '/datamaps.json')
        self.id2item = datamaps['id2item']
        self.item2id = datamaps['item2id']
        self.id2user = datamaps['id2user']
        self.cold_start_ratio = cold_start_ratio
        self.user_cold_start = user_cold_start
        self.interactions_left = cold_start_n_interact
        self.user_ids = list(self.sequential_data.keys())
        # Randomly select users for cold start based on the ratio
        self.cold_start_users = set(random.sample(self.user_ids, int(len(self.user_ids) * self.cold_start_ratio)))
        if augment:
            # Load augmentation data
            self.item_aug_data = load_json(data_path + f'/{aug_prefix}_augment.item')
            self.hist_aug_data = load_json(data_path + f'/{aug_prefix}_augment.hist')
            if user_cold_start:
                self.hist_aug_data_cs = load_json(data_path + f'/{aug_prefix}_augment_cs.hist')
                # Create a blended hist_aug_data
                for user_id in self.cold_start_users:
                    self.hist_aug_data[user_id] = self.hist_aug_data_cs[user_id]

    def __len__(self):
        return self.length

    def __getitem__(self, _id):
        
        if self.task == 'ctr':
            uid, seq_idx, lb = self.data[_id]
            item_seq, rating_seq = self.sequential_data[str(uid)]
            iid = item_seq[seq_idx]
            hist_seq_len = seq_idx - max(0, seq_idx - self.max_hist_len)
            attri_id = self.item2attribution[str(iid)]
            hist_item_seq = item_seq[max(0, seq_idx - self.max_hist_len): seq_idx]
            hist_rating_seq = rating_seq[max(0, seq_idx - self.max_hist_len): seq_idx]
            hist_attri_seq = [self.item2attribution[str(idx)] for idx in hist_item_seq]

            # Modify history for cold start users
            if self.user_cold_start and str(uid) in self.cold_start_users:
                no_interaction_id = self.item2id['no_interaction_item']
                num_to_replace = max(0, len(hist_item_seq) - self.interactions_left)
                hist_item_seq[:num_to_replace] = [no_interaction_id] * num_to_replace
                hist_rating_seq[:num_to_replace] = [0] * num_to_replace
                hist_attri_seq[:num_to_replace] = [self.item2attribution[str(no_interaction_id)]] * num_to_replace

            out_dict = {
                'iid': torch.tensor(iid).long(),
                'aid': torch.tensor(attri_id).long(),
                'lb': torch.tensor(lb).long(),
                'hist_iid_seq': torch.tensor(hist_item_seq).long(),
                'hist_aid_seq': torch.tensor(hist_attri_seq).long(),
                'hist_rate_seq': torch.tensor(hist_rating_seq).long(),
                'hist_seq_len': torch.tensor(hist_seq_len).long()
            }
            if self.augment:
                item_aug_vec = self.item_aug_data[str(self.id2item[str(iid)])]
                hist_aug_vec = self.hist_aug_data[str(self.id2user[str(uid)])]
                out_dict['item_aug_vec'] = torch.tensor(item_aug_vec).float()
                out_dict['hist_aug_vec'] = torch.tensor(hist_aug_vec).float()
        elif self.task == 'rerank':
            uid, seq_idx, candidates, candidate_lbs = self.data[_id]
            candidates_attr = [self.item2attribution[str(idx)] for idx in candidates]
            item_seq, rating_seq = self.sequential_data[str(uid)]
            hist_seq_len = seq_idx - max(0, seq_idx - self.max_hist_len)
            hist_item_seq = item_seq[max(0, seq_idx - self.max_hist_len): seq_idx]
            hist_rating_seq = rating_seq[max(0, seq_idx - self.max_hist_len): seq_idx]
            hist_attri_seq = [self.item2attribution[str(idx)] for idx in hist_item_seq]
            out_dict = {
                'iid_list': torch.tensor(candidates).long(),
                'aid_list': torch.tensor(candidates_attr).long(),
                'lb_list': torch.tensor(candidate_lbs).long(),
                'hist_iid_seq': torch.tensor(hist_item_seq).long(),
                'hist_aid_seq': torch.tensor(hist_attri_seq).long(),
                'hist_rate_seq': torch.tensor(hist_rating_seq).long(),
                'hist_seq_len': torch.tensor(hist_seq_len).long()
            }
            if self.augment:
                item_aug_vec = [torch.tensor(self.item_aug_data[str(self.id2item[str(idx)])]).float()
                                for idx in candidates]
                hist_aug_vec = self.hist_aug_data[str(self.id2user[str(uid)])]
                out_dict['item_aug_vec_list'] = item_aug_vec
                out_dict['hist_aug_vec'] = torch.tensor(hist_aug_vec).float()
        else:
            raise NotImplementedError

        return out_dict


