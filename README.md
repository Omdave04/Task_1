import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Titanic_Dataset.csv")

print("Basic Info:\n")
print(df.info())
print("\nMissing Values:\n")
print(df.isnull().sum())
print("\nSample Data:\n")
print(df.head())

df_cleaned = df.copy()
df_cleaned.drop(columns='Cabin', inplace=True)
df_cleaned['Age'].fillna(df_cleaned['Age'].median(), inplace=True)
df_cleaned['Embarked'].fillna(df_cleaned['Embarked'].mode()[0], inplace=True)

print("\nRemaining Missing Values:\n")
print(df_cleaned.isnull().sum())

df_encoded = pd.get_dummies(df_cleaned, columns=['Sex', 'Embarked'], drop_first=True)

numerical_cols = ['Age', 'Fare']
scaler = StandardScaler()
df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])

plt.figure(figsize=(10, 5))
for i, col in enumerate(numerical_cols):
    plt.subplot(1, 2, i + 1)
    sns.boxplot(data=df_encoded, y=col)
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()

for col in numerical_cols:
    Q1 = df_encoded[col].quantile(0.25)
    Q3 = df_encoded[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_encoded = df_encoded[(df_encoded[col] >= lower_bound) & (df_encoded[col] <= upper_bound)]

print("\nCleaned & Preprocessed Data Sample:\n")
print(df_encoded.head())
