import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

reference_dir = os.path.join('/app/input/', 'ref')
prediction_dir = os.path.join('/app/input/', 'res')
score_dir = '/app/output/'

prediction = np.genfromtxt(os.path.join(prediction_dir, 'prediction'))
truth = np.genfromtxt(os.path.join(reference_dir, 'testing_label'))
with open(os.path.join(prediction_dir, 'metadata.json')) as f:
    duration = json.load(f).get('duration', -1)
    
prediction = prediction[1:]  # Skip header
truth = truth[1:]  # Skip header

print('Checking Accuracy')
accuracy = accuracy_score(truth, prediction)
print(f'Accuracy: {accuracy}')

scores = {
    'accuracy': float(accuracy)
}
print(f'Scores: {scores}')

with open(os.path.join(score_dir, 'scores.json'), 'w') as score_file:
    json.dump(scores, score_file)

print('Scoring complete')
