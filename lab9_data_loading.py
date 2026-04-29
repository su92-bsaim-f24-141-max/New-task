# Asad | BSAI 4C | Roll: 141
# Lab 9: Data Loading and Exploration

import pandas as pd

# 1. Load the dataset (Titanic train.csv)
dataset = pd.read_csv('train.csv')

# 2. Print top 5 rows
print("First 5 rows:")
print(dataset.head())

# 3. Print bottom 5 rows
print("\nLast 5 rows:")
print(dataset.tail())

# 4. Number of rows and columns
print(f"\nRows: {dataset.shape[0]}, Columns: {dataset.shape[1]}")

# 5. Check for null values in each column
print("\nNull values per column:")
print(dataset.isnull().sum())

# 6. Data types of all columns
print("\nData types:")
print(dataset.dtypes)
