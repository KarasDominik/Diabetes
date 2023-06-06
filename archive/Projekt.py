# Import necessary libraries
import random

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model, svm, tree
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeCV, Ridge
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix, r2_score, \
    mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Read data from our csv file
data = pd.read_csv("diabetes.csv")

# Display every column of our dataset
pd.set_option("display.max_columns", None)

# Create DataFrame from our data
df = pd.DataFrame(data)

# # Print information about our dataset including columns, datatypes and range
# print(df.info())

# # Check if there are any values missing
# print(df.isnull().sum())

# Replace 0 with avg
# First replace 0 with nan to count average
column = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age', 'DiabetesPedigreeFunction']
df[column] = df[column].replace(0, np.nan)

# Count average and replace nan with average in each column
for factor in df:
    avg = df[factor].mean()
    df[factor] = df[factor].fillna(avg)

# Describe data
# print(df.describe())

# # Data shape
# print(df.shape)

# # Pair plot
# sns.pairplot(data=df, hue="Outcome")
# plt.show()

# # Box plots for all variables
# df.plot(kind ='box', subplots = True, sharey = False, figsize = (10,6))
# plt.subplots_adjust(wspace=0.5)
# plt.show()

# # Histogram for outcome and age
# plt.hist(df[df['Outcome']==1]['Age'], bins=5)
# plt.title("Distribution of Age for Women who has Diabetes")
# plt.xlabel('Age')
# plt.ylabel('Frequency')
# plt.show()

# # Histogram for outcome and BMI
# plt.hist(df[df['Outcome'] == 1]['BMI'], bins=5)
# plt.title("Distribution of BMI for Women who has Diabetes")
# plt.xlabel('Body Mass Index')
# plt.ylabel('Frequency')
# plt.show()

# # Covariances
# print(df.cov())

# # Correlations
# print(df.corr())

# # Create a heat map
# plt.figure(figsize = (12, 10))
# sns.heatmap(df.corr(), annot=True)
# plt.show()

# Graphs

# # Pregnancy by age
# plt.scatter(df['Age'], df['Pregnancies'], color = "black", marker = "x")
# plt.xlabel("Age")
# plt.ylabel("Pregnancies")
# plt.title("Pregnancies by age")
# plt.figure()

# # Insulin by glucose
# sns.scatterplot(x="Glucose", y = "Insulin", data = df)
# plt.show()

# # Skin thickness by BMI
# plt.scatter(df['SkinThickness'], df['BMI'], color="magenta")
# plt.xlabel("SkinThickness")
# plt.ylabel("BMI")
# plt.title("Skin thickness by BMI")
# plt.show()

# # Glucose by outcome
# sns.boxplot(data=df, x='Outcome', y='Glucose')
# plt.title("Glucose level by outcome")
# plt.show()

# Transformation

# Normalisation
# scaler = MinMaxScaler()
# norm = scaler.fit_transform(df[['Insulin']].values)
# plt.plot(df['Insulin'])
# plt.title("NON Normalised")
# plt.figure()
# plt.plot(norm)
# plt.title("Normalised")
# plt.show()

# Standardization
# scale = StandardScaler()
# scaled_data = scale.fit_transform(df[['Glucose']].values.reshape(-1, 1))
# plt.hist(df['Glucose'], 100)
# plt.title("NON Standardised")
# plt.figure()
# print(df['Glucose'].mean())
# plt.hist(scaled_data, 100)
# plt.title("Standardised")
# plt.show()

# Linear transformation
# plt.plot(df['SkinThickness'])
# plt.title("Without Linear Transformation")
# plt.figure()
# newSkinThickness = df['SkinThickness'] / 10
# print(newSkinThickness.mean())
# plt.plot(newSkinThickness)
# plt.title("With Linear Transformation")
# plt.show()

x = df.drop(['Outcome'], axis=1)
y = df['Outcome']

# Logistic regression

# x_train, x_test, y_train, y_test = train_test_split(x.values, y.values, test_size=0.5)
# clf = LogisticRegression(max_iter=1000)
# clf.fit(x_train, y_train)
# print(clf.score(x_test, y_test))
# print("----------Classification report-----------")
# print(classification_report(y_train, y_test))
#
# # Exemplary patient
# pregnancies = 4
# glucose = 150
# bloodPressure = 75
# skinThickness = 32
# insulin = 120
# BMI = 35
# diabetesPedigreeFunction = 0.29
# age = 24
# patient = np.array([[pregnancies, glucose, bloodPressure, skinThickness,
#                      insulin, BMI, diabetesPedigreeFunction, age]])
# patient = pd.DataFrame(patient)
#
# prediction = clf.predict(patient)
# print("Prediction:", prediction)

# # Naive Bayes

# x_train, x_test, y_train, y_test = train_test_split(x.values, y.values, test_size=0.5)
# gnb = GaussianNB()
#
# y_pred = gnb.fit(x_train, y_train).predict(x_test)
# print("Number of mislabeled points out of a total %d points : %d" % (x_test.shape[0], (y_test != y_pred).sum()))
# print(classification_report(y_test, y_pred))
# cm = confusion_matrix(y_test, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# disp.plot()
#
# plt.show()

# # PCA
#
dataTarget = df['Outcome']

scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)
pca = PCA(n_components=2)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)
# print(scaled_data.shape)
# print(x_pca.shape)
# plt.figure(figsize=(8, 6))
# plt.scatter(x_pca[:, 0], x_pca[:, 1], c=dataTarget, cmap='prism')
# plt.xlabel('First Principle Component')
# plt.ylabel('Second Principle Component')
# plt.show()

# # KNN

# X_train, X_test, y_train, y_test = train_test_split(df, dataTarget, test_size=0.5, random_state=0)
# clf = KNeighborsClassifier(n_neighbors=7)
# clf = clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# print("Without PCA")
# print(classification_report(y_test, y_pred))
#
# X_train, X_test, y_train, y_test = train_test_split(x_pca, dataTarget, test_size=0.5, random_state=0)
# clf = KNeighborsClassifier(n_neighbors=7)
# clf = clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# print("PCA")
# print(classification_report(y_test, y_pred))

# SVM

# print the names of the 13 features (X)
# print("Features: ", x.columns)
# # print the label type of cancer('malignant' 'benign') (Y)
# print("Labels: ", data.target_names)
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(x.values, y.values,
test_size=0.3, random_state=109) # 70% training and 30% test
#Create a svm Classifier
clf = svm.SVC(kernel='poly')  # Linear Kernel
#Train the model using the training sets
clf.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)
# Model Accuracy: how often is the classifier correct?
print("----------Classification report-----------")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(8, 8))
sns.scatterplot(x=X_train[:, 0], y=X_train[:, 1], hue=y_train)
plt.title("SVM")
plt.show()

# Decision tree
#
# X_train,X_test,y_train,y_test=train_test_split(x.values, y.values,test_size=0.5,random_state=0)
# clf=tree.DecisionTreeClassifier(criterion='gini',max_depth=3)
# clf=clf.fit(X_train,y_train)
# fig = plt.figure(figsize=(20, 12))
# _ = tree.plot_tree(clf,
#  feature_names=x.columns,
# class_names=['0', '1'],
# filled=True)
# fig.savefig("decision_tree.png")
# plt.show()
#
# # Random forest classifier
#
# X_train,X_test,y_train,y_test=train_test_split(x.values,y.values,test_size=0.5,random_state=0)
# rf=RandomForestClassifier(max_depth=10,random_state=0)
# clf=rf.fit(X_train,y_train)
# y_pred=clf.predict(X_test)
# print(clf.score(X_test,y_test))
# print(confusion_matrix(y_test,y_pred))
#
# clf = KNeighborsClassifier(n_neighbors=7)
# clf = clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# print(clf.score(X_test, y_test))
# print(confusion_matrix(y_test, y_pred))

# Multi layer classifier

# X_train, X_test, y_train, y_test = train_test_split(x.values, y.values, test_size=0.5, random_state=0)
# clf = MLPClassifier(max_iter=550)
# clf = clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# print(clf.coefs_)
# print(clf.n_layers_)
# print(clf.n_outputs_)
# print(clf.score(X_test, y_test))
# print(confusion_matrix(y_test, y_pred))
