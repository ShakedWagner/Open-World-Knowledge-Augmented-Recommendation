import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np


class Dice(nn.Module):
    """
    activation function DICE in DIN
    """

    def __init__(self):
        super(Dice, self).__init__()
        self.alpha = nn.Parameter(torch.zeros((1,)))
        self.epsilon = 1e-9

    def forward(self, x):
        norm_x = (x - x.mean(dim=0)) / torch.sqrt(x.var(dim=0) + self.epsilon)
        p = torch.sigmoid(norm_x)
        x = self.alpha * x.mul(1 - p) + x.mul(p)
        return x


class MLP(nn.Module):
    """
    Multi-layer Perceptron
    """
    def __init__(self, fc_dims, input_dim, dropout):
        super(MLP, self).__init__()
        fc_layers = []
        for fc_dim in fc_dims:
            fc_layers.append(nn.Linear(input_dim, fc_dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(p=dropout))
            input_dim = fc_dim
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        return self.fc(x)


class MoE(nn.Module):
    """
    Mixture of Experts
    """
    def __init__(self, moe_arch, inp_dim, dropout):
        super(MoE, self).__init__()
        export_num, export_arch = moe_arch
        self.export_num = export_num
        self.gate_net = nn.Linear(inp_dim, export_num)
        self.export_net = nn.ModuleList([MLP(export_arch, inp_dim, dropout) for _ in range(export_num)])

    def forward(self, x):
        gate = self.gate_net(x).view(-1, self.export_num)  # (bs, export_num)
        gate = nn.functional.softmax(gate, dim=-1).unsqueeze(dim=1) # (bs, 1, export_num)
        experts = [net(x) for net in self.export_net]
        experts = torch.stack(experts, dim=1)  # (bs, expert_num, emb)
        out = torch.matmul(gate, experts).squeeze(dim=1)
        return out


class HEA(nn.Module):
    """
    hybrid-expert adaptor
    """
    def __init__(self, ple_arch, inp_dim, dropout):
        super(HEA, self).__init__()
        share_expt_num, spcf_expt_num, expt_arch, task_num = ple_arch
        self.share_expt_net = nn.ModuleList([MLP(expt_arch, inp_dim, dropout) for _ in range(share_expt_num)])
        self.spcf_expt_net = nn.ModuleList([nn.ModuleList([MLP(expt_arch, inp_dim, dropout)
                                                           for _ in range(spcf_expt_num)]) for _ in range(task_num)])
        self.gate_net = nn.ModuleList([nn.Linear(inp_dim, share_expt_num + spcf_expt_num)
                                   for _ in range(task_num)])

    def forward(self, x_list):
        gates = [net(x) for net, x in zip(self.gate_net, x_list)]
        gates = torch.stack(gates, dim=1)  # (bs, tower_num, expert_num), export_num = share_expt_num + spcf_expt_num
        gates = nn.functional.softmax(gates, dim=-1).unsqueeze(dim=2)  # (bs, tower_num, 1, expert_num)
        cat_x = torch.stack(x_list, dim=1)  # (bs, tower_num, inp_dim)
        share_experts = [net(cat_x) for net in self.share_expt_net]
        share_experts = torch.stack(share_experts, dim=2)  # (bs, tower_num, share_expt_num, E)
        spcf_experts = [torch.stack([net(x) for net in nets], dim=1)
                        for nets, x in zip(self.spcf_expt_net, x_list)]
        spcf_experts = torch.stack(spcf_experts, dim=1)  # (bs, tower_num, spcf_expt_num, num)
        experts = torch.cat([share_experts, spcf_experts], dim=2)  # (bs, tower_num, expert_num, E)
        export_mix = torch.matmul(gates, experts).squeeze(dim=2)  # (bs, tower_num, E)
        # print('export mix', export_mix.shape, 'tower num', self.tower_num)
        export_mix = torch.split(export_mix, dim=1, split_size_or_sections=1)
        out = [x.squeeze(dim=1) for x in export_mix]
        return out


class ConvertNet(nn.Module):
    """
    convert from semantic space to recommendation space
    """
    def __init__(self, args, inp_dim, dropout, conv_type):
        super(ConvertNet, self).__init__()
        self.type = conv_type
        self.device = args.device
        print(self.type)
        if self.type == 'MoE':
            print('convert module: MoE')
            moe_arch = args.export_num, args.convert_arch
            self.sub_module = MoE(moe_arch, inp_dim, dropout)
        elif self.type == 'HEA':
            print('convert module: HEA')
            ple_arch = args.export_num, args.specific_export_num, args.convert_arch, args.augment_num
            self.sub_module = HEA(ple_arch, inp_dim, dropout)
        else:
            print('convert module: MLP')
            self.sub_module = MLP(args.convert_arch, inp_dim, dropout).to(self.device)

    def forward(self, x_list):
        if self.type == 'HEA':
            out = self.sub_module(x_list)
        else:
            out = [self.sub_module(x) for x in x_list]
        out = torch.cat(out, dim=-1)
        return out


class AttentionPoolingLayer(nn.Module):
    """
      attention pooling in DIN
    """

    def __init__(self, embedding_dim, dropout, fc_dims=[32, 16]):
        super(AttentionPoolingLayer, self).__init__()
        fc_layers = []
        input_dim = embedding_dim * 4
        # fc layer
        for fc_dim in fc_dims:
            fc_layers.append(nn.Linear(input_dim, fc_dim))
            fc_layers.append(Dice())
            fc_layers.append(nn.Dropout(p=dropout))
            input_dim = fc_dim

        fc_layers.append(nn.Linear(input_dim, 1))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, query, user_behavior, mask=None):
        """
          :param query_ad:embedding of target item   -> (bs, dim)
          :param user_behavior:embedding of user behaviors     ->  (bs, seq_len, dim)
          :param mask:mask on user behaviors  ->  (bs,seq_len, 1)
          :return output:user interest (bs, dim)
        """
        query = query.unsqueeze(1)
        seq_len = user_behavior.shape[1]
        queries = torch.cat([query] * seq_len, dim=1)
        attn_input = torch.cat([queries, user_behavior, queries - user_behavior,
                                queries * user_behavior], dim=-1)
        attns = self.fc(attn_input)
        if mask is not None:
            attns = attns.mul(mask)
        out = user_behavior.mul(attns)
        output = out.sum(dim=1)
        return output, attns

