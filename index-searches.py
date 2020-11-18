import pandas as pd
import helpers
import json
import numpy as np
import random
import math
import pickle
from sklearn.metrics.pairwise import linear_kernel


TEST = False
FACTORS = 100
EPOCHS = 25
ITEMS_PER_SEARCH = 5
ITEM_CANDIDATES_PER_USER = 30

print('Loading items...')
items_dict = helpers.load_items()
items_vectorizer, transformed_items, documents_to_item = helpers.vectorize_items(items_dict)
print('Loading interactions...')
interactions_train = helpers.load_interactions_df()
interactions_test = helpers.load_interactions_test_df()


def _normalize(item_title):
    return item_title.upper().strip().replace('.', '').replace('_', '').replace('?', '')

def get_search_queries(df):
    return list(set([_normalize(x).strip() for x in df[df['event_type'] == 'search']['item_id'].dropna().unique()]))

print('Generating search queries...')
train_search_queries = get_search_queries(interactions_train)
test_search_queries = get_search_queries(interactions_test)
search_queries = list(set(test_search_queries + train_search_queries))
sorted_search_queries = sorted(search_queries)

print('Starting...')

def process_search(text):
    query = items_vectorizer.transform([text]).astype(np.float32)
    results = linear_kernel(transformed_items, query)
    return [documents_to_item[x] for x in np.argsort(results.flatten())[-ITEMS_PER_SEARCH:][::-1]]

from multiprocessing import Pool

step = 10000
def process(interval):
    start, end = interval
    indexed_results = {}
    print(f'Started processing range [{start}-{end}]....')
    for i in range(start, end):
        q = sorted_search_queries[i]
        indexed_results[q] = process_search(q)
    print(f'Saving pickle [{start}-{end}] with {len(indexed_results)} elements...')
    with open(f'./data/search/search[{start}-{end}].pickle', 'wb') as handle:
        pickle.dump(indexed_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Saved pickle!')
            
with Pool(4) as p:
    p.map(process, [(x, x+step) for x in range(0, len(sorted_search_queries), step)])