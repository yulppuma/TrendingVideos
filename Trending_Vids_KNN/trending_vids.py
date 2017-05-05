import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import math
import os

test_cols = ['user id', 'date_posted', 'recent_views', 'total_views']
test = pd.read_csv('../data/test_trending (3).csv', sep=",", names=test_cols, encoding='latin-1')
train_cols = ['user id', 'date_posted', 'recent_views', 'total_views']
train = pd.read_csv('../data/train_trending(3).csv', sep=",", names=train_cols, encoding='latin-1')
#print(test.head())
#print(train.head())
col_recent_test = test.recent_views
col_date_test = test.date_posted
#print(col_recent_test[0])

test_recent_total = 0;
for i in range(len(col_recent_test)):
    test_recent_total += col_recent_test[i]
test_recent_total/=(len(col_recent_test)-1)
print(test_recent_total)
earliest_date_test = None
latest_date_test = None
test_date_total = 0
for i in range(len(col_date_test)):
    str = col_date_test[i]
    index = col_date_test[i].find("/", 3)
    num = int(str[index+1:])
    if i == 0:
        earliest_date_test = num
        latest_date_test = num
    test_date_total+=num
    if num < earliest_date_test:
        earliest_date_test = num
    if num > latest_date_test:
        latest_date_test = num
print(earliest_date_test)
print(latest_date_test)
trending_test_yrs = latest_date_test - (latest_date_test - earliest_date_test)*.25
print(trending_test_yrs)