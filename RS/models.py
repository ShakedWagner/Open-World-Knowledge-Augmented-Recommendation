import torch
import torch.nn as nn
from layers import AttentionPoolingLayer, MLP, ConvertNet
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss


def tau_function(x):
    return torch.where(x > 0, torch.exp(x), torch.zeros_like(x))


def attention_score(x, temperature=1.0):
    return tau_function(x / temperature) / (tau_function(x / temperature).sum(dim=1, keepdim=True) + 1e-20)


class BaseModel(nn.Module):
    """
    Base class for Recsys backbones - implemting Adapters for Knowledge Augmentation
    """
    def __init__(self, args, dataset):
        super(BaseModel, self).__init__()
        self.task = args.task
        self.args = args
        self.augment_num = 2 if args.augment else 0
        args.augment_num = self.augment_num

        self.item_num = dataset.item_num
        self.attr_num = dataset.attr_num
        self.attr_fnum = dataset.attr_ft_num
        self.rating_num = dataset.rating_num
        self.dense_dim = dataset.dense_dim
        self.max_hist_len = args.max_hist_len


        self.embed_dim = args.embed_dim
        self.final_mlp_arch = args.final_mlp_arch
        self.dropout = args.dropout
        self.hidden_size = args.hidden_size
        self.rnn_dp = args.rnn_dp
        self.output_dim = args.output_dim
        self.convert_dropout = args.convert_dropout
        self.convert_type = args.convert_type
        self.auxiliary_loss_weight = args.auxi_loss_weight

        self.item_fnum = 1 + self.attr_fnum
        self.hist_fnum = 2 + self.attr_fnum
        self.itm_emb_dim = self.item_fnum * self.embed_dim
        self.hist_emb_dim = self.hist_fnum * self.embed_dim
        self.dens_vec_num = 0

        self.item_embedding = nn.Embedding(self.item_num + 1, self.embed_dim)
        self.attr_embedding = nn.Embedding(self.attr_num + 1, self.embed_dim)
        self.rating_embedding = nn.Embedding(self.rating_num + 1, self.embed_dim)
        if self.augment_num:
            self.convert_module = ConvertNet(args, self.dense_dim, self.convert_dropout, self.convert_type)
            self.dens_vec_num = args.convert_arch[-1] * self.augment_num

        self.module_inp_dim = self.get_input_dim()
        self.field_num = self.get_field_num()
        self.convert_loss = 0

    def process_input(self, inp):
        device = next(self.parameters()).device
        hist_item_emb = self.item_embedding(inp['hist_iid_seq'].to(device)).view(-1, self.max_hist_len, self.embed_dim)
        hist_attr_emb = self.attr_embedding(inp['hist_aid_seq'].to(device)).view(-1, self.max_hist_len,
                                                                                 self.embed_dim * self.attr_fnum)
        hist_rating_emb = self.rating_embedding(inp['hist_rate_seq'].to(device)).view(-1, self.max_hist_len,
                                                                                      self.embed_dim)
        hist_emb = torch.cat([hist_item_emb, hist_attr_emb, hist_rating_emb], dim=-1)
        hist_len = inp['hist_seq_len'].to(device)

        if self.task == 'ctr':
            iid_emb = self.item_embedding(inp['iid'].to(device))
            attr_emb = self.attr_embedding(inp['aid'].to(device)).view(-1, self.embed_dim * self.attr_fnum)
            item_emb = torch.cat([iid_emb, attr_emb], dim=-1)
            # item_emb = item_emb.view(-1, self.itm_emb_dim)
            labels = inp['lb'].to(device)
            if self.augment_num:
                orig_dens_vec = [inp['hist_aug_vec'].to(device), inp['item_aug_vec'].to(device)]
                dens_vec = self.convert_module(orig_dens_vec)
            else:
                dens_vec, orig_dens_vec = None, None
            return item_emb, hist_emb, hist_len, dens_vec, orig_dens_vec, labels
        elif self.task == 'rerank':
            iid_emb = self.item_embedding(inp['iid_list'].to(device))
            attr_emb = self.attr_embedding(inp['aid_list'].to(device)).view(-1, self.max_list_len,
                                                                            self.embed_dim * self.attr_fnum)
            item_emb = torch.cat([iid_emb, attr_emb], dim=-1)
            item_emb = item_emb.view(-1, self.max_list_len, self.itm_emb_dim)
            labels = inp['lb_list'].to(device).view(-1, self.max_list_len)
            if self.augment_num:
                hist_aug = inp['hist_aug_vec'].to(device)
                item_list_aug = inp['item_aug_vec_list']
                orig_dens_list = [[hist_aug, item_aug.to(device)] for item_aug in item_list_aug]
                dens_vec_list = [self.convert_module(orig_dens) for orig_dens in orig_dens_list]
                dens_vec = torch.stack([dens for dens in dens_vec_list], dim=1)
            else:
                dens_vec, orig_dens_list = None, None

            return item_emb, hist_emb, hist_len, dens_vec, orig_dens_list, labels
        else:
            raise NotImplementedError

    def get_input_dim(self):
        if self.task == 'ctr':
            return self.hist_emb_dim + self.itm_emb_dim + self.dens_vec_num
        elif self.task == 'rerank':
            return self.itm_emb_dim + self.dens_vec_num
        else:
            raise NotImplementedError

    def get_field_num(self):
        return self.item_fnum + self.augment_num + self.hist_fnum

    def get_filed_input(self, inp):
        item_embedding, user_behavior, hist_len, dens_vec, orig_dens_vec, labels = self.process_input(inp)
        user_behavior = torch.mean(user_behavior, dim=1).view(-1, self.hist_emb_dim)
        if self.augment_num:
            inp = torch.cat([item_embedding, user_behavior, dens_vec], dim=1)
        else:
            inp = torch.cat([item_embedding, user_behavior], dim=1)
        out = inp.view(-1, self.field_num, self.embed_dim)
        return out, labels

    def process_rerank_inp(self, inp):
        item_embedding, user_behavior, hist_len, dens_vec, orig_dens_list, labels = self.process_input(inp)

        if self.augment_num:
            out = torch.cat([item_embedding, dens_vec], dim=-1)
        else:
            out = item_embedding
        return out, labels

    def get_ctr_output(self, logits, labels=None):
        outputs = {
            'logits': torch.sigmoid(logits),
            'labels': labels,
        }

        if labels is not None:
            if self.output_dim > 1:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view((-1, self.output_dim)), labels.float())
            else:
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits.view(-1), labels.view(-1).float())
            outputs['loss'] = loss + self.convert_loss * self.auxiliary_loss_weight

        return outputs

    def get_rerank_output(self, logits, labels=None, attn=False):
        outputs = {
            'logits': logits,
            'labels': labels,
        }

        if labels is not None:
            if attn:
                logits = attention_score(logits.view(-1, self.max_list_len), self.args.temperature)
                labels = attention_score(labels.float().view(-1, self.max_list_len), self.args.temperature)
            loss_fct = nn.BCELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1).float())
            outputs['loss'] = loss + self.convert_loss * self.auxiliary_loss_weight
        return outputs

    def get_mask(self, length, max_len):
        device = next(self.parameters()).device
        rang = torch.arange(0, max_len).view(-1, max_len).to(device)
        batch_rang = rang.repeat([length.shape[0], 1])
        mask = batch_rang < torch.unsqueeze(length, dim=-1)
        return mask.unsqueeze(dim=-1).long()


class DeepInterestNet(BaseModel):
    """
    DIN - Deep Interest Network for Click-Through Rate Prediction
    Implementation of the architecture described in the paper:
    https://arxiv.org/abs/1706.06978
    """

    def __init__(self, args, dataset):
        super(DeepInterestNet, self).__init__(args, dataset)

        self.map_layer = nn.Linear(self.hist_emb_dim, self.itm_emb_dim)
        # embedding of history item and candidate item should be the same
        self.attention_net = AttentionPoolingLayer(self.itm_emb_dim, self.dropout)

        # history embedding, item embedding, and user embedding
        self.final_mlp = MLP(self.final_mlp_arch, self.module_inp_dim, self.dropout)
        self.final_fc = nn.Linear(self.final_mlp_arch[-1], 1)

    def get_input_dim(self):
        return self.itm_emb_dim * 2 + self.dens_vec_num

    def forward(self, inp):
        """
            :param behaviors (bs, hist_len, hist_fnum)
            :param item_ft (bs, itm_fnum)
            :param user_ft (bs, usr_fnum)
            :return score (bs)
        """
        query, user_behavior, hist_len, dens_vec, orig_dens_vec, labels = self.process_input(inp)
        mask = self.get_mask(hist_len, self.max_hist_len)

        user_behavior = self.map_layer(user_behavior)
        user_interest, _ = self.attention_net(query, user_behavior, mask)

        if self.augment_num:
            concat_input = torch.cat([user_interest, query, dens_vec], dim=-1)
        else:
            concat_input = torch.cat([user_interest, query], dim=-1)

        mlp_out = self.final_mlp(concat_input)
        logits = self.final_fc(mlp_out)
        out = self.get_ctr_output(logits, labels)
        return out
