import json
import torch

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def save_json(data, file_path):
    with open(file_path, 'w') as w:
        json.dump(data, w)


def get_paragraph_representation(outputs, mask, pooler='cls', dim=1):
    """
    Get the paragraph representation from the encoder outputs in knwoledge encoding.
    """
    last_hidden = outputs.last_hidden_state
    hidden_states = outputs.hidden_states

    # Apply different poolers

    if pooler == 'cls':
        # There is a linear+activation layer after CLS representation
        return outputs.pooler_output.cpu()  # chatglm不能用，用于bert
    elif pooler == 'cls_before_pooler':
        return last_hidden[:, 0].cpu()
    elif pooler == "avg":
        return ((last_hidden * mask.unsqueeze(-1)).sum(dim) / mask.sum(dim).unsqueeze(-1)).cpu()
    elif pooler == "avg_first_last":
        first_hidden = hidden_states[1]
        last_hidden = hidden_states[-1]
        pooled_result = ((first_hidden + last_hidden) / 2.0 * mask.unsqueeze(-1)).sum(dim) / mask.sum(dim).unsqueeze(-1)
        return pooled_result.cpu()
    elif pooler == "avg_top2":
        second_last_hidden = hidden_states[-2]
        last_hidden = hidden_states[-1]
        pooled_result = ((last_hidden + second_last_hidden) / 2.0 * mask.unsqueeze(-1)).sum(dim) / mask.sum(dim).unsqueeze(-1)
        return pooled_result.cpu()
    elif pooler == 'len_last':  # 根据padding方式last方式也不一样
        lens = mask.unsqueeze(-1).sum(dim)
        # index = torch.arange(last_hidden.shape[0])
        # print(index)
        pooled_result = [last_hidden[i, lens[i] - 1, :] for i in range(last_hidden.shape[0])]
        pooled_result = torch.concat(pooled_result, dim=0)
        return pooled_result.cpu()
    elif pooler == 'last':
        if dim == 0:
            return last_hidden[-1, :, :]
        else:
            return last_hidden[:, -1, :]
    elif pooler == 'wavg':
        # Get weights of shape [bs, seq_len, hid_dim]
        weights = (
            torch.arange(start=1, end=last_hidden.shape[1] + 1)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(last_hidden.size())
            .float().to(last_hidden.device)
        )

        # Get attn mask of shape [bs, seq_len, hid_dim]
        input_mask_expanded = (
            mask
            .unsqueeze(-1)
            .expand(last_hidden.size())
            .float()
        )

        # Perform weighted mean pooling across seq_len: bs, seq_len, hidden_dim -> bs, hidden_dim
        sum_embeddings = torch.sum(last_hidden * input_mask_expanded * weights, dim=dim)
        sum_mask = torch.sum(input_mask_expanded * weights, dim=dim)

        pooled_result = sum_embeddings / sum_mask
        return pooled_result.cpu()
    else:
        raise NotImplementedError

