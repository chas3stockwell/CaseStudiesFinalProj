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

sample_students['diff'] = sample_students['high school gpa'] - sample_students['college gpa'] 

print(sample_students['college gpa'].mean)
print(sample_students['high school gpa'].mean)
print(sample_students['diff'].mean())

p_i = (sample_students['parental income'] - sample_students['parental income'].mean())/sample_students['parental income'].std()
y_vals = sample_students['college gpa']
print(sample_students['college gpa'][0])
sample_students['parental level of education'] = sample_students['parental level of education'].map({'some high school': 0, 'high school': 1, "some college":2, "associate's degree":3, "bachelor's degree":4, "master's degree":5})
print(sample_students)
print(sample_students.isnull().values.sum()) #No null vals
print(sample_students.iloc[:,:].columns.tolist())
correlations = sample_students.corr()
print(correlations['college gpa'].sort_values(ascending=False)) #Every value has some correlation besides years to graduate
sample_students = sample_students.drop(columns='years to graduate')

#fig_1 = plt.figure(figsize=(12,10))
#new_correlation = sample_students.corr()
#sns.heatmap(new_correlation, annot=True, cmap='Greens', annot_kws={'size':8})
#plt.title('College GPA Correlation Matrix')
#plt.show()

#highly_correlated_features = new_correlation[new_correlation > 0.75].fillna('-')
#print(highly_correlated_features)

#Plot the multivariate linreg model
sample_students = sample_students.drop(columns=['college gpa', 'parental income'])
sample_students.iloc[:,:] = (sample_students - sample_students.mean())/sample_students.std()
print(sample_students)

X = sample_students
ones = np.ones([len(sample_students), 1])
X = np.concatenate((ones, X), axis=1)
theta = np.zeros([1, len(sample_students.columns)+1])


y_vals = np.expand_dims(y_vals, axis=1)
target = y_vals
print(X.shape, target.shape, theta.shape)
#X_train, X_test, y_train, y_test = train_test_split(sample_students, y_vals)
#model = LinearRegression()
#model.fit(X_train, y_train)

def computecost(X, y,theta):
    H = X @ theta.T
    J = np.power((H - y), 2)
    sum = np.sum(J)/(2 * len(X))
    return sum

alpha = 0.01
iterations = 500

def gradientdescent(X, y,theta, iterations, alpha):
    cost = np.zeros(iterations)
    for i in range(iterations):
        H = X @ theta.T
        theta = theta - (alpha/len(X)) * np.sum(X * (H - y), axis=0)
        cost[i] = computecost(X, y,theta)
    return theta, cost

final_theta, cost = gradientdescent(X, y_vals,theta, iterations, alpha)
final_theta.round(2)

final_cost = computecost(X, y_vals,final_theta)
print(final_cost.round(3))

#Print Iterations vs. Cost
#fig_2, ax = plt.subplots(figsize=(10, 8))
#ax.plot(np.arange(iterations), cost, 'r')
#ax.set_xlabel('Iterations')
#ax.set_ylabel('Cost')
#ax.set_title('Iterations vs. Cost')
#plt.show()

def rmse(target, final_theta):
    predictions = X @ final_theta.T
    return np.sqrt(((predictions[:, 0] - target[:, 0]) ** 2).mean())

# Compute and display Root Mean Squared Error
rmse_val = rmse(target, final_theta)
print("me")
print(rmse_val.round(3))

predictions = X @ final_theta.T
print(str(predictions[1].round(3)))

sample_students['predicted_college_gpa'] = predictions.round(2)
sample_students['predicted_college_gpa'] = (sample_students['predicted_college_gpa'] - sample_students['predicted_college_gpa'].mean()) / sample_students['predicted_college_gpa'].std()
print(sample_students)



data = list(zip(sample_students['predicted_college_gpa'], p_i))
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
print(p_i)
kmeans = KMeans(n_clusters=4)
kmeans.fit(data)
print(sample_students['predicted_college_gpa'])
groups = ['']
plt.scatter(p_i, sample_students['predicted_college_gpa'], c=kmeans.labels_, )
plt.title("K Means Cluster for Student Demographics")
plt.xlabel('Parental Income')
plt.ylabel('Predicted College GPA')

y_locs, y_labels = plt.yticks()
y_locs = [ -2, -1, 0, 1, 2]
y_labels = [2.8, 3.1, 3.4, 3.7, 4.0]
plt.yticks(y_locs, y_labels)

locs, labels = plt.xticks()
print(labels)
locs = [-3, -2, -1, 0, 1, 2, 3]
labels = [30000, 45000, 60000, 75000, 90000, 105000, 120000]
plt.xticks(locs, labels)


#plt.show()

#Thus, a prediction model. 

