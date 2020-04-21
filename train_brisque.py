import glob
import pdb
import pickle
import time

import cv2
import numpy as np
from joblib import dump, load
from scipy import io
from sklearn import metrics, preprocessing
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVR
from tqdm import tqdm

from brisque import brisque

# load live dataset
print("load LIVE dataset")
files = glob.glob('dataset/LIVE/jpeg/*.bmp')
with open("dataset/LIVE/jpeg/info.txt") as f:
    lines = f.readlines()
file_scores = {}
for line in lines:
    filename = line.split(' ')[1]
    score = eval(line.split(' ')[2])
    file_scores[filename] = score
X = []
y = []
begin = time.time()
for f in tqdm(files):
    im = cv2.imread(f)
    filename = f.split('/')[-1]
    score = file_scores[filename]
    X.append(brisque(im))
    y.append(score)
end = time.time()
print("load data elapsed:{:.2}".format(end-begin))


X = np.array(X)
y = np.array(y)
print(X.shape)

train_x, test_x, train_y, test_y = train_test_split(
    X, y, test_size=0.05, random_state=42)

print("begin training...")
begin = time.time()
param_grid = [
    {'C': [1, 10, 100], 'gamma': [
        0.01, 0.001, 0.0001], 'kernel': ['rbf']},
]
clf = GridSearchCV(
    SVR(), param_grid, scoring='neg_mean_squared_error'
)
# clf = SVR()
clf.fit(train_x, train_y)
end = time.time()
print("training elapsed:{:.2}".format(end-begin))

print("begin test...")
# clf = load('svr_brisque.joblib')
pred_y = clf.predict(test_x)
for i in range(10):
    print(pred_y[i], test_y[i])
mse = metrics.mean_squared_error(test_y, pred_y)
print("mse:", mse)

print("model persistence...")
dump(clf, 'svr_brisque.joblib')
