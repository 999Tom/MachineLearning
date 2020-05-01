####import statements
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
data = pd.read_csv("data1.csv", sep=",")
#drawing the data(uncomment it to show the graph)
#%matplotlib inline
'''plt.plot(data.interac_time, data.Grade,'ro')
plt.xlabel('click times')
plt.show()'''
#one feature
hs_ratio = data[['interac_time']].copy()
#two feature
hs_ratio2 = data.drop(['Grade','Ratio'], axis = 1)
#split the trainitng and test set
x_train, x_test, y_train, y_test = train_test_split(hs_ratio2, data.Grade, test_size = 0.2)
x_train2, x_test2, y_train2, y_test2 = train_test_split(hs_ratio, data.Grade, test_size = 0.2)

#model Regression with different parameter(un common to see impact with these two commented parameters)
#two features
modelR1_F2 = LogisticRegression(solver = "newton-cg")
#modelR2_F2 = LogisticRegression(solver = "sag")
#modelR3_F2 = LogisticRegression(solver = "saga")
modelR4_F2 = LogisticRegression(solver = "lbfgs")
#one features
modelR1_F1 = LogisticRegression(solver = "newton-cg")
#modelR2_F1 = LogisticRegression(solver = "sag")
#modelR3_F1 = LogisticRegression(solver = "saga")
modelR4_F1 = LogisticRegression(solver = "lbfgs")

import time

start1 = time.time()#counting regression used time
#fitting the data in the model
modelR1_F2.fit(x_train,y_train)
#modelR2_F2.fit(x_train,y_train)
#modelR3_F2.fit(x_train,y_train)
modelR4_F2.fit(x_train,y_train)

modelR1_F1.fit(x_train2,y_train2)
#modelR2_F1.fit(x_train2,y_train2)
#modelR3_F1.fit(x_train2,y_train2)
modelR4_F1.fit(x_train2,y_train2)

#see the score
print(modelR1_F2.score(x_test,y_test))
#print(modelR2_F2.score(x_test,y_test))
#print(modelR3_F2.score(x_test,y_test))
print(modelR4_F2.score(x_test,y_test))

print(modelR1_F1.score(x_test2,y_test2))
#print(modelR2_F1.score(x_test2,y_test2))
#print(modelR3_F1.score(x_test2,y_test2))
print(modelR4_F1.score(x_test2,y_test2))
end1 = time.time()
timeused1 = end1-start1
print("time",timeused1)

### SVM start here
from sklearn import svm
modelSVM1_F2 = svm.SVC(kernel='linear') # Linear Kernel
modelSVM2_F2 = svm.SVC(kernel='rbf') # rbf Kernel
modelSVM3_F2 = svm.SVC(kernel='sigmoid') # sigmoid Kernel
modelSVM4_F2 = svm.SVC(kernel='poly') # poly Kernel


modelSVM1_F1 = svm.SVC(kernel='linear') # Linear Kernel
modelSVM2_F1 = svm.SVC(kernel='rbf') # Linear Kernel
modelSVM3_F1 = svm.SVC(kernel='sigmoid') # Linear Kernel
modelSVM4_F1 = svm.SVC(kernel='poly') # Linear Kernel

start2 = time.time()
#Train the model using the training sets
#fit the data in the model
modelSVM1_F2.fit(x_train, y_train)
modelSVM2_F2.fit(x_train, y_train)
modelSVM3_F2.fit(x_train, y_train)
modelSVM4_F2.fit(x_train, y_train)

modelSVM1_F1.fit(x_train2, y_train2)
modelSVM2_F1.fit(x_train2, y_train2)
modelSVM3_F1.fit(x_train2, y_train2)
modelSVM4_F1.fit(x_train2, y_train2)
#print the score
print(modelSVM1_F2.score(x_test, y_test))
print(modelSVM2_F2.score(x_test, y_test))
print(modelSVM3_F2.score(x_test, y_test))
print(modelSVM4_F2.score(x_test, y_test))

print(modelSVM1_F1.score(x_test2, y_test2))
print(modelSVM2_F1.score(x_test2, y_test2))
print(modelSVM3_F1.score(x_test2, y_test2))
print(modelSVM4_F1.score(x_test2, y_test2))
end2 = time.time()
timeused2 = end2-end1
print("time ",timeused2)
print("")
#print the report out
print("Logistic Regression used time: ",timeused1)
print("SVM used time: ",timeused2)
print("##2 features:##")
print("Logistic Regression:\n")

print("newton-cg: ",modelR1_F2.score(x_test,y_test))
#print("sag: ",modelR2_F2.score(x_test,y_test))
#print("saga: ",modelR3_F2.score(x_test,y_test))
print("lbfgs: ",modelR4_F2.score(x_test,y_test))
print("")
print("SVM:\n")
print("linear: ",modelSVM1_F2.score(x_test, y_test))
print("rbf: ",modelSVM2_F2.score(x_test, y_test))
print("sigmoid: ",modelSVM3_F2.score(x_test, y_test))
print("poly: ",modelSVM4_F2.score(x_test, y_test))
print("")
print("##1 feature:##")
print("Logistic Regression:\n")
print("newton-cg: ",modelR1_F1.score(x_test2,y_test2))
#print("sag: ",modelR2_F1.score(x_test2,y_test2))
#print("saga: ",modelR3_F1.score(x_test2,y_test2))
print("lbfgs: ",modelR4_F1.score(x_test2,y_test2))
print("")
print("SVM:\n")
print("linear: ",modelSVM1_F1.score(x_test2, y_test2))
print("rbf: ",modelSVM2_F1.score(x_test2, y_test2))
print("sigmoid: ",modelSVM3_F1.score(x_test2, y_test2))
print("poly: ",modelSVM4_F1.score(x_test2, y_test2))
