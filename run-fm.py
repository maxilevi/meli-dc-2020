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

TEST = True
FACTORS = 125
EPOCHS = 20
SEARCH_WEIGHT = 0.1# / 5
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
    
def build_candidate_pairs(users, valid_item_ids):
    users_column = []
    items_column = []
    user_lengths = []
    i = 0
    for u in users:
        candidates = [x for x in get_candidates(u) if x in valid_item_ids]
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


from rankfm.rankfm import RankFM

model = RankFM(factors=FACTORS, loss='warp', max_samples=20, alpha=0.01, sigma=0.1, learning_rate=0.10, learning_schedule='invscaling')

print(f"Fitting {interactions.shape[0]} interactions...")

model.fit(
    interactions,
    epochs=EPOCHS,
    verbose=True,
    sample_weight=sample_weights,
    #item_features=item_features,
    #user_features=user_features
)

import pickle

with open(f'./data/model_(e=25, t=False, sw=0.15, vw=3, t=st, f=125).pickle', 'wb') as f:
    pickle.dump(model, f)


print(f"Generating candidate pairs")
pairs, users_column, items_column, user_lengths = build_candidate_pairs(validation_users, valid_item_ids)

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


if not TEST and not user_target_dict:
    user_target_dict = interactions_train.groupby('user_id')['target'].unique().apply(lambda x: x).to_dict()


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

if TEST:
    submit = pd.DataFrame(recommendations)
    print(f'Submit shape is {submit.shape}')
    assert submit.shape == (10, 177070)
    submit.transpose().to_csv(f'final_submit.csv', index=False, header=False)

#if TEST:
#    import pickle
#    with open(f"./data/recommendations/E={epochs}", "wb") as f:
#        pickle.dump(recommendations_pairs, f)