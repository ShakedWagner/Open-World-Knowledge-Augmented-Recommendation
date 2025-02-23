import numpy as np
import torch
import torch.utils.data as Data
from utils import load_json, load_pickle
import random
from sklearn.mixture import GaussianMixture

class KARDataset(Data.Dataset):
    def __init__(self, data_path, set_name='train', task='ctr', max_hist_len=10, augment=False, aug_prefix=None, user_cold_start=False, cold_start_ratio=0.0, cold_start_n_interact=1, cs_method='demographic'):
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
        self.user_ids = [int(uid) for uid in self.sequential_data.keys()]
        self.set_uids = torch.unique(torch.tensor([x[0] for x in self.data])).tolist()
        self.train_uids = self.set_uids if set_name == 'train' else list(set(self.user_ids) - set(self.set_uids))
        # Randomly select users for cold start based on the ratio
        self.cold_start_users = set(random.sample(self.set_uids, int(len(self.set_uids) * self.cold_start_ratio)))
        self.cs_method = cs_method
        if augment:
            # Load augmentation data
            self.item_aug_data = load_json(data_path + f'/{aug_prefix}_augment.item')
            self.hist_aug_data = load_json(data_path + f'/{aug_prefix}_augment.hist')

            if user_cold_start:
                # load pre encoded demographic data
                if self.cs_method == 'demographic':
                    self.hist_aug_data_cs = load_json(data_path + f'/{aug_prefix}_augment_cs.hist')
                # sample representative embeddings from the training set gmm
                elif self.cs_method == 'gmm':
                    self.hist_aug_data_to_sample = torch.tensor([v for k,v in self.hist_aug_data.items() if int(k) not in self.cold_start_users and int(k) in self.train_uids])
                    self.hist_aug_data_cs = sample_from_embeddings_gmm(self.hist_aug_data_to_sample, n_components=5, n_samples=len(self.cold_start_users))
                    self.hist_aug_data_cs = {str(k):v for k,v in zip(self.cold_start_users, self.hist_aug_data_cs)}
                # replace the embeddings with zeros
                elif self.cs_method == 'zeros':
                    self.hist_aug_data_cs = {str(k):torch.zeros(len(self.hist_aug_data[str(k)])) for k in self.cold_start_users}
                else:
                    raise NotImplementedError(f"Cold start method {self.cs_method} not implemented choose from demographic or gmm instead")
                
                # Create a blended hist_aug_data
                for user_id in self.cold_start_users:
                    self.hist_aug_data[str(user_id)] = self.hist_aug_data_cs[str(user_id)]

    def __len__(self):
        return self.length

    def __getitem__(self, _id):
        
        uid, seq_idx, lb = self.data[_id]
        item_seq, rating_seq = self.sequential_data[str(uid)]
        iid = item_seq[seq_idx]
        hist_seq_len = seq_idx - max(0, seq_idx - self.max_hist_len)
        attri_id = self.item2attribution[str(iid)]
        hist_item_seq = item_seq[max(0, seq_idx - self.max_hist_len): seq_idx]
        hist_rating_seq = rating_seq[max(0, seq_idx - self.max_hist_len): seq_idx]
        hist_attri_seq = [self.item2attribution[str(idx)] for idx in hist_item_seq]

        # Modify history for cold start users
        if self.user_cold_start and uid in self.cold_start_users:
            # Replace the last num_to_replace interactions with no_interaction_item to simulate cold start
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


        return out_dict


def sample_from_embeddings_gmm(embeddings, n_components=5, n_samples=10):
    """
    Samples new vectors from a Gaussian Mixture Model fitted to the embeddings.

    Parameters:
    - embeddings: array-like, shape (n_samples, n_features)
      The original embeddings.
    - n_components: int
      The number of mixture components in the GMM.
    - n_samples: int
      The number of new samples to generate.

    Returns:
    - samples: array, shape (n_samples, n_features)
      The new sampled embeddings.
    """
    embeddings = np.array(embeddings)
    
    # Fit a Gaussian Mixture Model to the embeddings
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(embeddings)
    
    # Sample new embeddings from the GMM
    samples, _ = gmm.sample(n_samples)
    
    return samples