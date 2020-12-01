import pandas as pd
import helpers
import json
import numpy as np
import math
import pickle
from sklearn.metrics.pairwise import linear_kernel
import collections
import gc

print('Loading interactions...')
interactions_train = helpers.load_interactions_unprocessed_df()
interactions_test = helpers.load_interactions_unprocessed_test_df()

print('Loading search map')
with open(f'./data/search_map.pickle', 'rb') as handle:
    search_map = pickle.load(handle)
print('Loading items')
items_dict = helpers.load_items()
print('Starting processing...')


def process_search_rows(df):
    items_counter = collections.defaultdict(int)
    new_rows = collections.defaultdict(list)
    i = 0

    def process(row):
        nonlocal i
        i += 1
        if row['event_type'] != 'search' or type(row['item_id']) != str:
            return row
        str_id = helpers._normalize(row['item_id']).strip()
        items = search_map[str_id]
        
        item = items[items_counter[str_id]]
        items_counter[str_id] = (items_counter[str_id] + 1) % 5
        row['item_id'] = item
        
        if i % 100000 == 0:
            print(f"{i} processed")

        return row
    
    n_df = df
    for i in range(5):
        n_df = n_df.append(df[df['event_type'] == 'search'])
        print(f'Appended {i}, {n_df.shape}')
    n_df = n_df.sort_values(by=['user_id', 'event_timestamp']).reset_index(drop=True)
    print(f'Sorted')
    n_df = n_df.apply(process, axis=1)
    return n_df


step = 2000000
for j in range(0, interactions_train.shape[0], step):
    p = interactions_train[j:j+step]
    print(f'Processing {p.shape}')
    interactions_train_partial = process_search_rows(p)
    interactions_train_partial.to_csv(f'./data/interactions_train_complete_{j}.csv', index=False)
    print(f'Processed train [{j},{j+step}]')
    gc.collect()


for j in range(0, interactions_test.shape[0], step):
    p = interactions_test[j:j+step]
    print(f'Processing {p.shape}')
    interactions_test_partial = process_search_rows(p)
    interactions_test_partial.to_csv(f'./data/interactions_test_complete_{j}.csv', index=False)
    print(f'Processed test [{j},{j+step}]')
    gc.collect()