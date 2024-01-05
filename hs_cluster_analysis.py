from random import sample
import pandas as pd
from pandas import DataFrame
import seaborn as sns
from matplotlib import pyplot as plt
import math
import scipy.stats
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans





sample_students = pd.read_csv("graduation_rate.csv")
sample_students = sample_students.drop(columns='parental level of education')
sample_students.iloc[:,:] = (sample_students - sample_students.mean())/sample_students.std()
print(sample_students.columns)


data = list(zip(sample_students['high school gpa'], sample_students['parental income']))
inertias = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

#plt.plot(range(1,11), inertias, marker='o')
#plt.title('Elbow method')
#plt.xlabel('Number of clusters')
#plt.ylabel('Inertia')
#plt.show()

#ax1 = sample_students.plot.scatter(x='SAT total score',
#                      y='high school gpa',
#                      c='blue')
#plt.show()

kmeans = KMeans(n_clusters=4)
kmeans.fit(data)

groups = ['']
plt.scatter(sample_students['parental income'], sample_students['college gpa'], c=kmeans.labels_, )
plt.title("K Means Cluster for Student Demographics")
plt.xlabel('Parental Income')
plt.ylabel('Predicted College GPA')
y_locs, y_labels = plt.yticks()
y_locs = [-3, -2, -1, 0, 1]
y_labels = [2.8, 3.1, 3.4, 3.7, 4.0]
plt.yticks(y_locs, y_labels)

locs, labels = plt.xticks()
print(labels)
locs = [-3, -2, -1, 0, 1, 2, 3]
labels = [30000, 45000, 60000, 75000, 90000, 105000, 120000]
plt.xticks(locs, labels)


plt.show()