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
FACTORS = 100
EPOCHS = 20
SEARCH_WEIGHT = 0.15# / 5
VIEW_WEIGHT = 3
KAGGLE = False
MODEL_TYPE = 'rankfm'

items = helpers.load_items_df()
items_dict = helpers.load_items()
domain_item_dict = helpers.load_domain_item_dict(items_dict)
all_items = list(items_dict.keys())
print("Items loaded!")
interactions_train = helpers.load_interactions_df()
if TEST:
    interactions_test = helpers.load_interactions_test_df()
print("Interactions loaded!")

def encode_interactions(df):
    new_df = df[pd.notnull(df['item_id'])].copy()
    new_df['user_id'] = new_df['user_id'].astype(float).astype(int)
    new_df['item_id'] = new_df['item_id'].astype(float).astype(int)
    sample_weights = np.array([(VIEW_WEIGHT if x != 'search' else SEARCH_WEIGHT) for x in new_df['event_type']])
    return new_df[['user_id', 'item_id']], sample_weights
    
def build_candidate_pairs(users, valid_item_ids, use_valid_ids):
    users_column = []
    items_column = []
    user_lengths = []
    i = 0
    for u in users:
        candidates = [x for x in get_candidates(u) if x in valid_item_ids or not use_valid_ids]
        items_column += candidates
        users_column += [u] * len(candidates)
        user_lengths.append((u, len(candidates)))
        if i % 100000 == 0:
            print(f"Progress {i}/{len(users)}")
        i += 1
    pairs = pd.DataFrame({'user_id': users_column, 'item_id': items_column})
    return pairs, users_column, items_column, user_lengths

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


def get_domains_from_items(items):
    return set(items_dict[int(item)]['domain_id'] for item in items)

def get_candidates(user):
    items_interacted = event_dict[user] if user in event_dict else set()

    domains = get_domains_from_items(items_interacted) if items_interacted else top_domains[:10]
    items_for_domains = [domain_top_items[d] for d in domains]
    item_universe = sum(items_for_domains, [])

    for item in item_universe:
        items_interacted.add(item)
            
    return list(items_interacted)


def combine_interactions(i1, i2):
    i1c = i1.copy()
    i2c = i2.copy()
    i2c['user_id'] += i1c.shape[0]
    return i1c.append(i2c)


users = None
interactions = None
sample_weights = None
user_features = None

if TEST:
    interactions = (combine_interactions(interactions_train, interactions_test))
    validation_users = interactions_test.user_id.unique() + interactions_train.shape[0]
    all_users = np.concatenate([interactions_train.user_id.unique(), validation_users])
else:
    interactions = (interactions_train)
    validation_users = interactions_train.user_id.unique()
    all_users = validation_users

user_target_dict = None

## Calculate auxiliary data
interactions, sample_weights = encode_interactions(interactions)
domain_top_items = helpers.load_top_items(interactions_train, domain_item_dict)
top_domains = helpers.load_top_domains(interactions_train, domain_top_items)
event_dict = interactions.groupby('user_id')['item_id'].unique().apply(set).to_dict()
valid_item_ids = set(interactions['item_id'].unique())

import pickle

print(f"Generating candidate pairs")
pairs, users_column, items_column, user_lengths = build_candidate_pairs(validation_users, valid_item_ids, True)

with open(f'./data/pairs-small.pickle', 'wb') as f:
    pickle.dump({'pairs': pairs, 'users_column': users_column, 'items_column': items_column, 'user_lengths': user_lengths}, f)

print(f"Generating candidate pairs")
pairs, users_column, items_column, user_lengths = build_candidate_pairs(validation_users, valid_item_ids, False)

with open(f'./data/pairs-big.pickle', 'wb') as f:
    pickle.dump({'pairs': pairs, 'users_column': users_column, 'items_column': items_column, 'user_lengths': user_lengths}, f)