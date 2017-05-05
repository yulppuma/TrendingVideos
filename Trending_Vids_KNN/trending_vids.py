import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import math
import os

test_cols = ['user id', 'date_posted', 'recent_views', 'total_views']
test = pd.read_csv('../data/test_trending (3).csv', sep=",", names=test_cols, encoding='latin-1')
train_cols = ['user id', 'date_posted', 'recent_views', 'total_views']
train = pd.read_csv('../data/train_trending(3).csv', sep=",", names=train_cols, encoding='latin-1')
#Set columns of recent views and date poated to col_recent_test & col_date_test, respectively
col_recent_test = np.array(test.recent_views)
col_date_test = np.array(test.date_posted)
#Getting average of recent views from all 10000+ videos
test_recent_total = 0;
#Don't know why I had this, but I think I want to find average of posted video dates
test_date_total = 0
#Initializing earliest and latest posted video
earliest_date_test = None
latest_date_test = None
for i in range(len(col_recent_test)):
    test_recent_total += col_recent_test[i]
    str = col_date_test[i]
    index = col_date_test[i].find("/", 3)
    num = int(str[index + 1:])
    col_date_test[i] = num
    if i == 0:
        earliest_date_test = num
        latest_date_test = num
    test_date_total += num
    if num < earliest_date_test:
        earliest_date_test = num
    if num > latest_date_test:
        latest_date_test = num
test_recent_total/=(len(col_recent_test)-1)
print(test_recent_total)
print(col_date_test[0])
print(earliest_date_test)
print(latest_date_test)
trending_test_yrs = latest_date_test - (latest_date_test - earliest_date_test)*.25
print(trending_test_yrs)
X = np.array((col_date_test, col_recent_test)).T
x = [1,2,3]
y = [4,5,6]
z = zip(x,y)
print(list(z))
nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors([[2092, 49999]])
print (indices)