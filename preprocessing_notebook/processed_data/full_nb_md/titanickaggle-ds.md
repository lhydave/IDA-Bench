```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```


```python
# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load Titanic dataset
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')

# Create copies to avoid SettingWithCopyWarning
train_df = train_df.copy()
test_df = test_df.copy()

# ---- Step 1: Exploratory Data Analysis (EDA) ----
# Check for missing values
print("Missing values in train dataset:\n", train_df.isnull().sum())

# Visualizations
plt.figure(figsize=(8, 5))
sns.countplot(x='Survived', data=train_df, palette='coolwarm')
plt.title('Survival Distribution')
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(x='Pclass', hue='Survived', data=train_df, palette='coolwarm')
plt.title('Survival Rate by Passenger Class')
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x='Survived', y='Age', data=train_df, palette='coolwarm')
plt.title('Age Distribution by Survival')
plt.show()

# ---- Step 2: Data Preprocessing (Fully Fixed) ----
# Fill missing Age values based on Pclass median
train_df.loc[:, "Age"] = train_df.groupby("Pclass")["Age"].transform(lambda x: x.fillna(x.median()))
test_df.loc[:, "Age"] = test_df.groupby("Pclass")["Age"].transform(lambda x: x.fillna(x.median()))

# Extract first letter of Cabin as a new feature
train_df.loc[:, "CabinLetter"] = train_df["Cabin"].str[0].fillna("X")  # X = no cabin info
test_df.loc[:, "CabinLetter"] = test_df["Cabin"].str[0].fillna("X")

# Fill missing Embarked values with the most common value (Fixed Warning)
train_df.loc[:, "Embarked"] = train_df["Embarked"].fillna(train_df["Embarked"].mode()[0])
test_df.loc[:, "Embarked"] = test_df["Embarked"].fillna(test_df["Embarked"].mode()[0])

# Fill missing Fare values based on Pclass median
test_df.loc[:, "Fare"] = test_df.groupby("Pclass")["Fare"].transform(lambda x: x.fillna(x.median()))

# Create a new feature: FamilySize
train_df.loc[:, "FamilySize"] = train_df["SibSp"] + train_df["Parch"] + 1
test_df.loc[:, "FamilySize"] = test_df["SibSp"] + test_df["Parch"] + 1

# Create a new feature: IsAlone (1 = alone, 0 = with family)
train_df.loc[:, "IsAlone"] = (train_df["FamilySize"] == 1).astype(int)
test_df.loc[:, "IsAlone"] = (test_df["FamilySize"] == 1).astype(int)

# Convert categorical variables to numerical
label_encoder = LabelEncoder()
train_df.loc[:, "Sex"] = label_encoder.fit_transform(train_df["Sex"])
test_df.loc[:, "Sex"] = label_encoder.transform(test_df["Sex"])

train_df.loc[:, "Embarked"] = label_encoder.fit_transform(train_df["Embarked"])
test_df.loc[:, "Embarked"] = label_encoder.transform(test_df["Embarked"])

train_df.loc[:, "CabinLetter"] = label_encoder.fit_transform(train_df["CabinLetter"])
test_df.loc[:, "CabinLetter"] = label_encoder.transform(test_df["CabinLetter"])

# Drop unnecessary columns
train_df.drop(columns=["Name", "Ticket", "Cabin"], inplace=True)
test_df.drop(columns=["Name", "Ticket", "Cabin"], inplace=True)

# Define features and target variable
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "CabinLetter", "FamilySize", "IsAlone"]
X = train_df[features]
y = train_df["Survived"]

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- Step 3: Model Training (Improved) ----
# Tune Random Forest hyperparameters using GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring="accuracy")
grid_search.fit(X_train, y_train)

# Best model
best_rf = grid_search.best_estimator_

# Make predictions on the validation set
y_pred = best_rf.predict(X_val)

# Evaluate model performance
accuracy = accuracy_score(y_val, y_pred)
print("Validation Accuracy:", accuracy)

# Confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_val, y_pred), annot=True, fmt='d', cmap='coolwarm')
plt.title("Confusion Matrix")
plt.show()

# ---- Step 4: Generate Predictions for Submission ----
X_test = test_df[features]
predictions = best_rf.predict(X_test)

# Create submission file
submission = pd.DataFrame({"PassengerId": test_df["PassengerId"], "Survived": predictions})
submission.to_csv("submission.csv", index=False)

# Display confirmation message
print("✅ Submission file saved successfully! Upload 'submission.csv' to Kaggle.")

```
