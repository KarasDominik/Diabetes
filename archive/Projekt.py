# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Read data from our csv file
data = pd.read_csv("diabetes.csv")

# Display every column of our dataset
pd.set_option("display.max_columns", None)

# Create DataFrame from our data
df = pd.DataFrame(data)

# Print information about our dataset including columns, datatypes and range
# print(df.info())

# Check if there are any values missing
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

# Data shape
# print(df.shape)

# Covariances
# print(df.cov())

# Correlations
# print(df.corr())

# Create a heat map
# plt.figure(figsize = (12, 10))
# sns.heatmap(df.corr(), annot=True)
# plt.show()

# Graphs

# Pregnancy by age
# plt.scatter(df['Age'], df['Pregnancies'], color = "black", marker = "x")
# plt.xlabel("Age")
# plt.ylabel("Pregnancies")
# plt.title("Pregnancies by age")
# plt.figure()

# Outcome by glucose
# df.boxplot(column='Glucose', by='Outcome')
# plt.show()


# Skin thickness by BMI
# plt.scatter(df['SkinThickness'], df['BMI'], color="magenta")
# plt.xlabel("SkinThickness")
# plt.ylabel("BMI")
# plt.title("Skin thickness by BMI")
# plt.show()

# Transformation

# Normalisation
scaler = MinMaxScaler()
norm = scaler.fit_transform(df[['Insulin']].values)
plt.plot(df['Insulin'])
plt.title("NON Normalised")
plt.figure()
plt.plot(norm)
plt.title("Normalised")
plt.show()

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
