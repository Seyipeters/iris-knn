# Iris KNN Classification

This project implements a complete machine learning pipeline for classifying Iris flower species using the K-Nearest Neighbors (KNN) algorithm.

## Dataset

The dataset used is `iris_with_missing.csv`, which includes measurements of sepal length, sepal width, petal length, petal width, and the species label. The dataset contains some missing values that are handled during preprocessing.

## Pipeline Overview

1. **Data Preprocessing**:
   - Load and describe the dataset.
   - Check and impute missing values (mean for numerical, mode for categorical).
   - Encode categorical target variable.
   - Apply feature scaling (standardization).

2. **Data Splitting**:
   - Stratified sampling to split into 70% training, 15% validation, 15% test sets.

3. **Model Training & Hyperparameter Tuning**:
   - Train initial KNN model with k=3.
   - Tune hyperparameters (k: 3,5,7; distance: Euclidean, Manhattan) using validation set.

4. **Model Evaluation**:
   - Evaluate on test set: accuracy, precision, recall, F1-score, ROC-AUC.
   - Perform leave-one-out cross-validation for robustness assessment.

## Insights

- The Iris dataset is small and well-suited for KNN, which performed well.
- Standardization of features is crucial for distance-based algorithms like KNN.
- Hyperparameter tuning on the validation set helped optimize performance.
- Leave-one-out CV provided a reliable estimate of model generalization.

## Potential Improvements

- Experiment with other algorithms (e.g., SVM, Decision Trees, Random Forest).
- Feature selection or engineering to reduce dimensionality.
- Collect more data or use data augmentation.
- Implement ensemble methods for better performance.
- Add visualization for decision boundaries.

## Requirements

- Python 3.x
- Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn

## Usage

Run the `iris_knn.ipynb` notebook to execute the pipeline.