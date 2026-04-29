# Asad | BSAI 4C | Roll: 141
# Lab 10: Data Pre-processing (Dealing with nulls, changing dtypes)

import pandas as pd
import numpy as np

# Load dataset
data = pd.read_csv('train.csv')

print("Original shape:", data.shape)
print("Null values before cleaning:")
print(data.isnull().sum())

# Fill missing Age with median
data['Age'].fillna(data['Age'].median(), inplace=True)

# Fill missing Embarked with mode (most frequent)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Drop Cabin column (too many missing values)
data.drop('Cabin', axis=1, inplace=True)

# Verify no nulls remain
print("\nNull values after cleaning:")
print(data.isnull().sum())

# Convert object columns to integer codes
obj_cols = data.select_dtypes(include=['object']).columns
for col in obj_cols:
    data[col] = pd.factorize(data[col])[0]   # each unique string becomes an int

print("\nData types after conversion:")
print(data.dtypes)

# Preview cleaned data
print("\nCleaned data (first 5 rows):")
print(data.head())
