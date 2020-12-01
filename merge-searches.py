import os
import pickle
import pandas as pd

interactions_train = pd.DataFrame()

for file in os.listdir('./data/full/train/'):
    print(f'Loading {file}')
    interactions_train = interactions_train.append(pd.read_csv(f'./data/full/train/{file}'))

interactions_train.sort_values(by=['user_id', 'event_timestamp']).reset_index(drop=True).to_csv('./data/full/interactions_train_full.csv', index=False)

interactions_test = pd.DataFrame()

for file in os.listdir('./data/full/test/'):
    print(f'Loading {file}')
    interactions_test = interactions_test.append(pd.read_csv(f'./data/full/test/{file}'))

interactions_test.sort_values(by=['user_id', 'event_timestamp']).reset_index(drop=True).to_csv('./data/full/interactions_test_full.csv', index=False)
        