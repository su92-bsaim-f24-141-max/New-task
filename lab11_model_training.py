# Asad | BSAI 4C | Roll: 141
# Lab 11: Train/Test Split, Model Training, Evaluation & Saving Model

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib

# ---- 1. Load and preprocess (same as Lab 10) ----
script_dir = os.path.dirname(os.path.abspath(__file__))
data = pd.read_csv(os.path.join(script_dir, 'train.csv'))
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data.drop('Cabin', axis=1, inplace=True)

# Convert object columns to integer codes
obj_cols = data.select_dtypes(include=['object']).columns
for col in obj_cols:
    data[col] = pd.factorize(data[col])[0]

# ---- 2. Split features and target ----
X = data.drop('Survived', axis=1)   # all columns except target
y = data['Survived']

# ---- 3. Train-test split (70% train, 30% test) ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=141, shuffle=True
)

# ---- 4. Define classifiers ----
classifiers = {
    'Bernoulli': BernoulliNB(),
    'Random Forest': RandomForestClassifier(random_state=141),
    'Gaussian': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(random_state=141),
    'Multinomial': MultinomialNB(),
    'KNeighbors': KNeighborsClassifier()
}

# Store results
results = {'Classifier': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1': []}

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = metrics.accuracy_score(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = metrics.recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = metrics.f1_score(y_test, y_pred, average='weighted', zero_division=0)

    results['Classifier'].append(name)
    results['Accuracy'].append(acc)
    results['Precision'].append(prec)
    results['Recall'].append(rec)
    results['F1'].append(f1)

    print(f"{name}: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")

# ---- 5. Plot line graph of all metrics ----
plt.figure(figsize=(14, 8))
for metric in ['Accuracy', 'Precision', 'Recall', 'F1']:
    plt.plot(results['Classifier'], results[metric], marker='o', label=metric)
plt.title('Classifier Performance Comparison')
plt.xlabel('Classifier')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('lab11_line_chart.png')
print("\nLine chart saved as 'lab11_line_chart.png'")

# ---- 6. Plot bar chart of F1 scores ----
plt.figure(figsize=(10, 6))
bars = plt.bar(results['Classifier'], results['F1'],
               color=['#08737f', '#00898a', '#089f8f', '#39b48e', '#64c987', '#92dc7e'])
plt.xlabel('Classifiers')
plt.ylabel('F1 Score')
plt.title('F1 Scores of Applied Classifiers')
for bar, score in zip(bars, results['F1']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{score:.3f}', ha='center', va='bottom')
plt.tight_layout()
plt.savefig('lab11_bar_chart.png')
print("Bar chart saved as 'lab11_bar_chart.png'")

# ---- 7. Save the best model (Random Forest) for Lab 12 ----
best_model = RandomForestClassifier(random_state=141)
best_model.fit(X_train, y_train)
joblib.dump(best_model, 'titanic_model.pkl')
print("\nModel saved as 'titanic_model.pkl'")

# Also save the column order for later use in Flask
joblib.dump(X.columns.tolist(), 'model_columns.pkl')
print("Column order saved as 'model_columns.pkl'")
