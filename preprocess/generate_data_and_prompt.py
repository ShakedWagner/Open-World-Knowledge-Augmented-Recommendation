import os
from datetime import date
import random
from collections import defaultdict
from pre_utils import load_json, save_json, save_pickle, GENDER_MAPPING, \
    AGE_MAPPING, OCCUPATION_MAPPING

def generate_ctr_data(sequence_data, lm_hist_idx, uid_set, rating_threshold):
    """
    Generate CTR data for training and testing. Generate binary labels for each interaction 0 if rating is lower than rating_threshold, 1 otherwise.
    """
    # print(list(lm_hist_idx.values())[:10])
    full_data = []
    total_label = []
    for uid in uid_set:
        start_idx = lm_hist_idx[str(uid)]
        item_seq, rating_seq = sequence_data[str(uid)]
        for idx in range(start_idx, len(item_seq)):
            label = 1 if rating_seq[idx] > rating_threshold else 0
            full_data.append([uid, idx, label])
            total_label.append(label)
    print('user num', len(uid_set), 'data num', len(full_data), 'pos ratio',
          sum(total_label) / len(total_label))
    print(full_data[:5])
    return full_data


def generate_hist_prompt(sequence_data, item2attribute, datamap, lm_hist_idx, dataset_name):
    """
    Generate history prompt for language model.
    """
    itemid2title = datamap['itemid2title']
    attrid2name = datamap['id2attribute']
    id2user = datamap['id2user']
    if dataset_name == 'ml-1m':
        user2attribute = datamap['user2attribute']
    hist_prompts = {}
    print('item2attribute', list(item2attribute.items())[:10])
    for uid, item_rating in sequence_data.items():
        user = id2user[uid]
        item_seq, rating_seq = item_rating
        cur_idx = lm_hist_idx[uid]
        hist_item_seq = item_seq[:cur_idx]
        hist_rating_seq = rating_seq[:cur_idx]
        history_texts = []
        for iid, rating in zip(hist_item_seq, hist_rating_seq):
            tmp = '"{}", {} stars; '.format(itemid2title[str(iid)], int(rating))
            history_texts.append(tmp)
        if dataset_name == 'amz':
            # prompt = 'Analyze user\'s preferences on product (consider factors like genre, functionality, quality, ' \
            #          'price, design, reputation. Provide clear explanations based on ' \
            #          'relevant details from the user\'s product viewing history and other pertinent factors.'
            # hist_prompts[uid] = 'Given user\'s product rating history: ' + ''.join(history_texts) + prompt
            prompt = 'Analyze user\'s preferences on books about factors like genre, author, writing style, theme, ' \
                     'setting, length and complexity, time period, literary quality, critical acclaim (Provide ' \
                     'clear explanations based on relevant details from the user\'s book viewing history and other ' \
                     'pertinent factors.'
            hist_prompts[user] = 'Given user\'s book rating history: ' + ''.join(history_texts) + prompt
        elif dataset_name == 'ml-1m':
            gender, age, occupation = user2attribute[user]
            user_text = 'Given a {} user who is aged {} and {}, this user\'s movie viewing history over time' \
                        ' is listed below. '.format(GENDER_MAPPING[gender], AGE_MAPPING[age],
                                                    OCCUPATION_MAPPING[occupation])
            question = 'Analyze user\'s preferences on movies (consider factors like genre, director/actors, time ' \
                       'period/country, character, plot/theme, mood/tone, critical acclaim/award, production quality, ' \
                       'and soundtrack). Provide clear explanations based on relevant details from the user\'s movie ' \
                       'viewing history and other pertinent factors.'
            hist_prompts[user] = user_text + ''.join(history_texts) + question
        else:
            raise NotImplementedError
    print('data num', len(hist_prompts))
    print(list(hist_prompts.items())[0])
    return hist_prompts


def generate_item_prompt(item2attribute, datamap, dataset_name):
    """
    Generate item prompt for language model.
    """
    itemid2title = datamap['itemid2title']
    attrid2name = datamap['id2attribute']
    id2item = datamap['id2item']
    item_prompts = {}
    for iid, title in itemid2title.items():
        item = id2item[iid]
        if dataset_name == 'amz':
            brand, cate = item2attribute[str(iid)]
            brand_name = attrid2name[str(brand)]
            # cate_name = attrid2name[cate]
            item_prompts[item] = 'Introduce book {}, which is from brand {} and describe its attributes including but' \
                                ' not limited to genre, author, writing style, theme, setting, length and complexity, ' \
                                'time period, literary quality, critical acclaim.'.format(title, brand_name)
            # item_prompts[iid] = 'Introduce product {}, which is from brand {} and describe its attributes (including but' \
            #                     ' not limited to genre, functionality, quality, price, design, reputation).'.format(title, brand_name)
        elif dataset_name == 'ml-1m':
            item_prompts[item] = 'Introduce movie {} and describe its attributes (including but not limited to genre, ' \
                                'director/cast, country, character, plot/theme, mood/tone, critical ' \
                                'acclaim/award, production quality, and soundtrack).'.format(title)
        else:
            raise NotImplementedError
    print('data num', len(item_prompts))
    print(list(item_prompts.items())[0])
    return item_prompts


if __name__ == '__main__':
    random.seed(12345)
    BASE_DIR = r''
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    DATA_SET_NAME = 'ml-1m'
    rating_threshold = 3
    PROCESSED_DIR = os.path.join(DATA_DIR, DATA_SET_NAME, 'proc_data')
    SEQUENCE_PATH = os.path.join(PROCESSED_DIR, 'sequential_data.json')
    ITEM2ATTRIBUTE_PATH = os.path.join(PROCESSED_DIR, 'item2attributes.json')
    DATAMAP_PATH = os.path.join(PROCESSED_DIR, 'datamaps.json')
    SPLIT_PATH = os.path.join(PROCESSED_DIR, 'train_test_split.json')
    # load interactions in a list of [uid, iid, rating]
    sequence_data = load_json(SEQUENCE_PATH)
    train_test_split = load_json(SPLIT_PATH)
    # load item2attribute in a dict of {iid: [attribute1, attribute2, ...]}
    item2attribute = load_json(ITEM2ATTRIBUTE_PATH)
    item_set = list(map(int, item2attribute.keys()))
    print('final loading data')

    print('generating ctr train dataset')
    train_ctr = generate_ctr_data(sequence_data, train_test_split['lm_hist_idx'],
                                  train_test_split['train'], rating_threshold)
    print('generating ctr validation dataset')
    validation_ctr = generate_ctr_data(sequence_data, train_test_split['lm_hist_idx'],
                                       train_test_split['validation'], rating_threshold)
    print('generating ctr test dataset')
    test_ctr = generate_ctr_data(sequence_data, train_test_split['lm_hist_idx'],
                                 train_test_split['test'], rating_threshold)
    print('save ctr data')
    save_pickle(train_ctr, os.path.join(PROCESSED_DIR, 'ctr.train'))
    save_pickle(validation_ctr, os.path.join(PROCESSED_DIR, 'ctr.validation'))
    save_pickle(test_ctr, os.path.join(PROCESSED_DIR, 'ctr.test'))

   

    datamap = load_json(DATAMAP_PATH)
    # write stats to json
    statis = {
        
        'attribute_ft_num': datamap['attribute_ft_num'],
        'rating_threshold': rating_threshold,
        'item_num': len(datamap['id2item']),
        'attribute_num': len(datamap['id2attribute']),
        'rating_num': 5,
        'dense_dim': 0,
    }
    save_json(statis, os.path.join(PROCESSED_DIR, 'stat.json'))

    print('generating item prompt')
    item_prompt = generate_item_prompt(item2attribute, datamap, DATA_SET_NAME)
    print('generating history prompt')
    hist_prompt = generate_hist_prompt(sequence_data, item2attribute, datamap,
                                       train_test_split['lm_hist_idx'], DATA_SET_NAME)
    print('save prompt data')
    save_json(item_prompt, os.path.join(PROCESSED_DIR, 'prompt.item'))
    save_json(hist_prompt, os.path.join(PROCESSED_DIR, 'prompt.hist'))
    item_prompt, hist_prompt = None, None
