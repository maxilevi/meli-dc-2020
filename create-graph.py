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
import pickle
import networkx as nx

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
    
def build_candidate_pairs(users, valid_item_ids, bfs_cache, components):
    users_column = []
    items_column = []
    user_lengths = []
    i = 0
    for u in users:
        candidates = [x for x in get_candidates(u, bfs_cache, components) if x in valid_item_ids]
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

def get_candidates(user, bfs_cache, components):
    items_interacted = event_dict[user] if user in event_dict else set()
    its = set(sum([components[bfs_cache[int(x)]] for x in items_interacted], []))
    #domains = get_domains_from_items(its)
    #items_for_domains = [domain_top_items[d] for d in domains]
    #item_universe = set(sum(items_for_domains, []))
    return its#item_universe


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

import gc

print(f"Generating graph")
G = collections.defaultdict(set)
for j, user in enumerate(validation_users):
    items_interacted = list(event_dict[user] if user in event_dict else set())
    
    for i, d1 in enumerate(items_interacted):
        for d2 in items_interacted[i+1:]:
            v1 = int(d1)
            v2 = int(d2) 
            G[v1].add(v2)
            G[v2].add(v1)
    if j % 10000 == 0 and j > 0: 
        print(f'{j}/{len(validation_users)}')

with open(f'./data/train-graph.pickle', 'wb') as f:
    pickle.dump(G, f)

gc.collect()

for v in G.keys():
    G[v] = list(G[v])

gc.collect()
print(f"Generating BFS map")
components = []
bfs_cache = {}
for v in G.keys():
    if v not in bfs_cache:
        visited = set()
        queue = collections.deque([v])
        while queue:
            print(len(queue))
            w = queue.pop()
            visited.add(w)
            for n in G[w]:
                if n in visited:
                    continue
                queue.appendleft(n)
        for w in visited:
            bfs_cache[w] = len(components)
        components.append(list(visited))
        print(f'Appeding components {len(visited)}')
    del G[v]

del G
gc.collect()

print(f"Generating candidate pairs")
pairs, users_column, items_column, user_lengths = build_candidate_pairs(validation_users, valid_item_ids, bfs_cache, components)

with open(f'./data/pairs-graph.pickle', 'wb') as f:
    pickle.dump({'pairs': pairs, 'users_column': users_column, 'items_column': items_column, 'user_lengths': user_lengths}, f)