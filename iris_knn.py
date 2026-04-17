import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('iris_with_missing.csv')

# Describe the dataset
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nSummary statistics:")
print(df.describe(include='all'))

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Handle missing values - impute with mean for numerical, mode for categorical
imputer_num = SimpleImputer(strategy='mean')
imputer_cat = SimpleImputer(strategy='most_frequent')

numerical_cols = ['sepal length', 'sepal width', 'petal length', 'petal width']
categorical_cols = ['class']

df[numerical_cols] = imputer_num.fit_transform(df[numerical_cols])
df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])

print("\nAfter imputation, missing values:")
print(df.isnull().sum())

# Encode categorical variables
le = LabelEncoder()
df['class'] = le.fit_transform(df['class'])

print("\nClasses:", le.classes_)

# Feature scaling - Standardization
scaler = StandardScaler()
X = df.drop('class', axis=1)
y = df['class']

X_scaled = scaler.fit_transform(X)

print("\nScaled features shape:", X_scaled.shape)

# Data Splitting - 70% train, 15% validation, 15% test with stratified sampling
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print("\nTrain set shape:", X_train.shape, y_train.shape)
print("Validation set shape:", X_val.shape, y_val.shape)
print("Test set shape:", X_test.shape, y_test.shape)

# Model Training - KNN with k=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Hyperparameter Tuning on validation set
k_values = [3, 5, 7]
metrics = ['euclidean', 'manhattan']
best_score = 0
best_params = {}

for k in k_values:
    for metric in metrics:
        knn_temp = KNeighborsClassifier(n_neighbors=k, metric=metric)
        knn_temp.fit(X_train, y_train)
        val_score = accuracy_score(y_val, knn_temp.predict(X_val))
        if val_score > best_score:
            best_score = val_score
            best_params = {'k': k, 'metric': metric}

print("\nBest hyperparameters:", best_params)
print("Best validation accuracy:", best_score)

# Train final model with best params
knn_final = KNeighborsClassifier(n_neighbors=best_params['k'], metric=best_params['metric'])
knn_final.fit(X_train, y_train)

# Evaluate on test set
y_pred = knn_final.predict(X_test)
y_pred_proba = knn_final.predict_proba(X_test)

print("\nTest Set Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("Recall:", recall_score(y_test, y_pred, average='macro'))
print("F1-Score:", f1_score(y_test, y_pred, average='macro'))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_proba, multi_class='ovr'))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Leave-One-Out Cross-Validation
loo = LeaveOneOut()
loo_scores = []

for train_index, test_index in loo.split(X_scaled):
    X_train_loo, X_test_loo = X_scaled[train_index], X_scaled[test_index]
    y_train_loo, y_test_loo = y.iloc[train_index], y.iloc[test_index]
    
    knn_loo = KNeighborsClassifier(n_neighbors=best_params['k'], metric=best_params['metric'])
    knn_loo.fit(X_train_loo, y_train_loo)
    loo_scores.append(accuracy_score(y_test_loo, knn_loo.predict(X_test_loo)))

print("\nLeave-One-Out CV Accuracy:", np.mean(loo_scores))

# Visualization - Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')
plt.show()