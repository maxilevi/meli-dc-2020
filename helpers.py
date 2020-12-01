import pickle
import pandas as pd
import os
import math
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

LOCAL_INTERACTIONS_PATH = './data/interactions_train_complete.csv'
LOCAL_INTERACTIONS_TEST_PATH = './data/interactions_test_complete.csv'
LOCAL_ITEMS_PICKLE = './data/items.pickle'

KAGGLE_INTERACTIONS_PATH = '../input/mercadolibre-dc/data_search/interactions_train_complete.csv'
KAGGLE_INTERACTIONS_TEST_PATH = '../input/mercadolibre-dc/data_search/interactions_test_complete.csv'
KAGGLE_ITEMS_PICKLE = '../input/mercadolibre-dc/items/items.pickle'

INTERACTIONS_PATH = LOCAL_INTERACTIONS_PATH if os.path.exists(LOCAL_INTERACTIONS_PATH) else KAGGLE_INTERACTIONS_PATH
INTERACTIONS_TEST_PATH = LOCAL_INTERACTIONS_TEST_PATH if os.path.exists(LOCAL_INTERACTIONS_TEST_PATH) else KAGGLE_INTERACTIONS_TEST_PATH
ITEMS_PICKLE = LOCAL_ITEMS_PICKLE if os.path.exists(LOCAL_ITEMS_PICKLE) else KAGGLE_ITEMS_PICKLE
ITEMS_PATH = './data/items.csv'

def _normalize(item_title):
    return (item_title.upper().strip().replace('.', '').replace('_', '').replace('?', '')).strip()

def vectorize_items(items_dict):
    documents_to_item = [x for x in items_dict.keys()]
    documents = [
        _normalize(items_dict[x]['title']) for x in items_dict.keys()
    ]
    vectorizer = TfidfVectorizer(stop_words=['spanish', 'portuguese'], strip_accents='unicode')
    X = vectorizer.fit_transform(documents)
    return vectorizer, X.astype(np.float32), documents_to_item

def load_interactions_df():
    return pd.read_csv(INTERACTIONS_PATH)

def load_interactions_test_df():
    return pd.read_csv(INTERACTIONS_TEST_PATH)

def load_interactions_full_df():
    return pd.read_csv('./data/interactions_train_full.csv')

def load_interactions_test_full_df():
    return pd.read_csv('./data/interactions_test_full.csv')

def load_interactions_unprocessed_df():
    return pd.read_csv('./data/interactions.csv')

def load_interactions_unprocessed_test_df():
    return pd.read_csv('./data/interactions_test.csv')

def load_items_df():
    return pd.read_csv(ITEMS_PATH)

def load_items():
    with open(ITEMS_PICKLE, 'rb') as handle:
        return pickle.load(handle)

def load_domain_item_dict(items_dict):
    domain_item_dict = {}
    for item in items_dict.keys():
        domain = items_dict[item]['domain_id']
        if domain not in domain_item_dict:
            domain_item_dict[domain] = []
        domain_item_dict[domain].append(item)
    return domain_item_dict

def load_top_domains(interactions, domain_top_items):
	item_counts = interactions['target'].value_counts()
	return list(sorted(domain_top_items.keys(), key=lambda x: -sum([item_counts.at[int(y)] if int(y) in item_counts.index else 0 for y in domain_top_items[x]])))

def load_top_items(interactions, domain_item_dict):
    counts = interactions['target'].value_counts()
    domain_top_items = {}
    for domain in domain_item_dict.keys():
        sorted_items = sorted([(x, counts.at[int(x)] if int(x) in counts.index else 0) for x in domain_item_dict[domain]], key=lambda x: -x[1])
        domain_top_items[domain] = [x[0] for x in sorted_items[:50]]
    return domain_top_items


def bin_array(n, m):
    return np.array(list(np.binary_repr(n).zfill(m))).astype(np.int8)

def pickle_items():
    item_dict = {}
    item_df = pd.read_json('./data/item_data.jl', lines=True, chunksize=500)
    def apply_row(line):
        item_dict[line.item_id] = {
            "title": line.title,
            "domain_id": line.domain_id,
            "product_id": line.product_id,
            "price": line.price,
            "category_id": line.category_id,
            "condition": line.condition,
        }

    for df in item_df:
        df.apply(apply_row, axis=1)

    print(f"Loaded {len(item_dict)} items into memory")

    with open(ITEMS_PICKLE, 'wb') as handle:
        pickle.dump(item_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _relevance(items_dict, item, target):
    if item == target:
        return 15
    elif items_dict[item]['domain_id'] == items_dict[target]['domain_id']:
        return 1
    return 0

def _get_perfect_dcg():
    perfect = [15, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    return sum(perfect[i] / np.log2(i + 2) for i in range(len(perfect))) / len(perfect)

def _dcg(items_dict, recommendations, target):
    
    dcg = sum(_relevance(items_dict, recommendations[i], target) / np.log2(i + 2) for i in range(len(recommendations)))
    return dcg / len(recommendations)

def ndcg_score(items_dict, recommendations, user_targets_dict):
    sum_ndcg = 0
    sum_perfect = 0
    for x in recommendations.keys():
        sum_ndcg += _dcg(items_dict, [int(w) for w in recommendations[x]], int(user_targets_dict[x]))
        sum_perfect += _get_perfect_dcg()

    return sum_ndcg / sum_perfect