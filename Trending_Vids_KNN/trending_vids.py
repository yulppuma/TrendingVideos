import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import math
import os
def getAverageRecent(column):
    # Getting average columns from all 10000+ videos
    test_recent_total = 0;
    for i in range(len(column)):
        test_recent_total += column[i]
    test_recent_total /= (len(column) - 1)
    return test_recent_total

def parseDates(column):
    # Change column format from mm/dd/yyyy to year(integer)
    for i in range(len(column)):
        str = column[i]
        index = column[i].find("/", 3)
        num = int(str[index + 1:])
        column[i] = num
    return column

def classifyVids(column, avg):
    #Check to see if video is trending by comparing it to avg value
    isTrending = np.array(column)
    for i in range(len(isTrending)):
        if column[i] > avg:
            isTrending[i] = 1
        else:
            isTrending[i] = 0
    return isTrending

def calculateError(isTrending, X):
    date=0
    views=0
    results=np.empty(len(isTrending))
    #performs KNN on each data point
    for i in range (len(X)):
        views=X[i][1]
        date=X[i][0]
        test_nbrs = NearestNeighbors(n_neighbors=100, algorithm='ball_tree').fit(X)
        distances, indices = test_nbrs.kneighbors([[date, views]])
        results[i]=KNNClassify(isTrending, indices) #fills in array with results from knn for each point

    numIncorrect=0
    percentError=0.0
    error=np.empty(len(isTrending))
    for i in range(len(isTrending)):
        if isTrending[i]==results[i]:
            error[i]=0 #0 if no error
        else:
            error[i]=1
    for i in range(len(error)):
        if error[i]==1:#if error
            numIncorrect+=1; #increment count of incorrect result
    percentError=(numIncorrect/len(isTrending)) #calculate total error value of data set
    return percentError

def KNNClassify(isTrending, indices): #returns 1 if classifies with trending
    output = 0
    count = 0
    index = 0
    for i in range(100):
        index=(indices[0][i])
        count+=isTrending[index]
    if count > 50: #if data point classifies with a majority of 1's
        output = 1
    else:#if data point classifies with a mojority of 0's
        output=0
    return output

test_cols = ['user id', 'date_posted', 'recent_views', 'total_views']
test = pd.read_csv('../data/test_trending (3).csv', sep=",", names=test_cols, encoding='latin-1')
train_cols = ['user id', 'date_posted', 'recent_views', 'total_views']
train = pd.read_csv('../data/train_trending(3).csv', sep=",", names=train_cols, encoding='latin-1')
#Set columns of recent views and date posted to col_recent_test & col_date_test, respectively
col_recent_test = np.array(test.recent_views)
col_date_test = np.array(test.date_posted)
col_recent_train = np.array(train.recent_views)
col_date_train = np.array(train.date_posted)

test_dates = parseDates(col_date_test)
train_dates = parseDates(col_date_train)

#Avg value of recent/dates test & train
avg_recent_test = getAverageRecent(col_recent_test)
print (avg_recent_test)
avg_recent_train = getAverageRecent(col_recent_train)
print (avg_recent_train)
avg_dates_test = getAverageRecent(test_dates)
print(avg_dates_test)
avg_dates_train = getAverageRecent(train_dates)
print(avg_dates_train)

#Initializing the data set that will be used for KNN algorithm
    #X = test data ; Y = train data
X = np.array((test_dates, col_recent_test)).T
test_nbrs = NearestNeighbors(n_neighbors=100, algorithm='ball_tree').fit(X)
distances, indices = test_nbrs.kneighbors([[2092, 49999]])

Y = np.array((train_dates, col_recent_train)).T
train_nbrs = NearestNeighbors(n_neighbors=100, algorithm='ball_tree').fit(Y)
distances_train, indices_train = train_nbrs.kneighbors([[2092, 49999]])

isTrending = classifyVids(col_recent_test, avg_recent_test)

print(calculateError(isTrending, X))
print(calculateError(isTrending, Y))
