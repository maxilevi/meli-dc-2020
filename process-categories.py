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
interactions_train = helpers.load_interactions_df()
interactions_test = helpers.load_interactions_test_df()
print('Loading items')
items_dict = helpers.load_items()
domain_item_dict = helpers.load_domain_item_dict(items_dict)
domain_map = {d:i for i, d in enumerate(domain_item_dict.keys())}
print('Starting processing...')


def process(row):
    if type(row['item_id']) != float or not math.isnan(row['item_id']):
        row['item_id'] = domain_map[items_dict[int(row['item_id'])]['domain_id']]
    if 'target' in row:
        row['target'] = domain_map[items_dict[int(row['target'])]['domain_id']]
    return row


print(f'Processing {interactions_train.shape}')
interactions_train_cats = interactions_train.apply(process, axis=1)
interactions_train_cats.to_csv(f'./data/interactions_train_cats.csv', index=False)
print(f'Processed train')


print(f'Processing {interactions_test.shape}')
interactions_test_cats = interactions_test.apply(process, axis=1)
interactions_test_cats.to_csv(f'./data/interactions_test_cats.csv', index=False)
print(f'Processed test')

import pickle
with open('./data/domain_map.pickle', 'wb') as f:
    pickle.dump(domain_map, f)