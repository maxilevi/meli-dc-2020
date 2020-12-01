import pandas as pd
import helpers
import json
import numpy as np
import random
import math
import pickle
from sklearn.preprocessing import MinMaxScaler
import collections
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

TEST = False
MODEL_PATH='./data/model_(e=20, t=False, sw=0.15, vw=3, t=st).pickle'
CANDIDATES_PATH='./data/pairs-small.pickle'

items_dict = helpers.load_items()
domain_item_dict = helpers.load_domain_item_dict(items_dict)
all_items = list(items_dict.keys())
print("Items loaded!")
interactions_train = helpers.load_interactions_df()
if TEST:
    interactions_test = helpers.load_interactions_test_df()
user_target_dict = interactions_train.groupby('user_id')['target'].unique().apply(lambda x: x).to_dict()

def build_recommendations(recommendations_pairs, items_column, user_lengths):
    offset = 0
    recommendations = {}
    for user, user_len in user_lengths:
        user_recs = recommendations_pairs[offset:offset+user_len]
        ranked_recs = np.argsort(user_recs)[::-1]
        top_10 = [x for x in ranked_recs if not np.isnan(user_recs[x])][:10]
        recommendations[user] = [items_column[x + offset] for x in top_10]
        offset += user_len
    return recommendations


if TEST:
    validation_users = interactions_test.user_id.unique() + interactions_train.shape[0]
else:
    validation_users = interactions_train.user_id.unique()

import pickle

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(CANDIDATES_PATH, 'rb') as f:
    pairs_data = pickle.load(f)

print(f"Loading candidate pairs")
print(len(pairs_data))
print(list(pairs_data.keys()))
pairs, users_column, items_column, user_lengths = (pairs_data['pairs'], pairs_data['users_column'], pairs_data['items_column'], pairs_data['user_lengths'])
print(len(pairs))

print(f"Generating recommnedation pairs")
recommendations_pairs = model.predict(pairs, cold_start='nan')
recommendations = build_recommendations(recommendations_pairs, items_column, user_lengths)

def fill(recommendations):
    for k in recommendations.keys():
        if len(recommendations[k]) == 0:
            recommendations[k] = random.choices(all_items, k=10)
        elif len(recommendations[k]) < 10:
            category = items_dict[recommendations[k][0]]['domain_id']
            recommendations[k] += random.choices(domain_item_dict[category], k=(10 - len(recommendations[k])))

# Assert required sizes
            
assert len(recommendations) == len(validation_users)
unfilled = len([True for k in recommendations.keys() if len(recommendations[k]) != 10])
if unfilled > 0:
    print(f"{unfilled} entries were not filled. Extending the items...")
    fill(recommendations)


def _relevance(items_dict, item, target):
    if item == target:
        return 15
    if items_dict[item]['domain_id'] == items_dict[target]['domain_id']:
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

if not TEST:
    print(ndcg_score(items_dict, recommendations, user_target_dict))