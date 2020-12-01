import pickle
import os

models = []
for file in os.listdir('./data/models/'):
	with open(file, 'rb') as f:
		print(f'Loading model {file}')
		models.append(pickle.load(f))

with open(file, 'rb') as f:
	pairs = pickle.load(f)

scores = []
for model in models:
	predictions = model.predict(pairs, cold_start='nan')
	if not scores:
		scores = predictions
	else:
		for i, n in enumerate(predictions):
			scores[i] += n
