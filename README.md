# APS Failure Detection with Tree-Based Models and SMOTE

This project involves classifying failure events in APS systems using tree-based models, focusing on handling class imbalance with SMOTE and model tuning. Techniques include random forests, XGBoost, and multivariate decision trees, with data pre-processing to manage missing values and imbalance. This project is part of Homework 6 for the DSCI 552 course.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Tree-Based Models](#tree-based-models)
- [SMOTE for Class Imbalance](#smote-for-class-imbalance)
- [Requirements](#requirements)

## Project Overview
The primary objectives of this project are:
1. To classify APS failure events using tree-based models, including random forests and XGBoost.
2. To explore methods for handling class imbalance, particularly with SMOTE.
3. To evaluate model performance using confusion matrix, ROC, AUC, and out-of-bag error estimates.

## Dataset
The APS Failure dataset can be downloaded from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/APS+Failure+at+Scania+Trucks). It contains:
- A training set with 60,000 instances (1,000 positive cases) and 171 columns (one class column).
- All features are numeric and may contain missing values.

## Data Preprocessing
1. **Imputation of Missing Values**: 
   - Selected imputation techniques to address missing values in the dataset.
2. **Feature Analysis**:
   - Calculated the Coefficient of Variation (CV) for each feature.
   - Created a correlation matrix to visualize relationships.
   - Visualized the features with the highest CV to understand their significance.

## Tree-Based Models
1. **Random Forest**:
   - Trained a random forest without compensating for class imbalance.
   - Calculated confusion matrix, ROC, AUC, and misclassification rates on both training and test sets.
   - Used Out-of-Bag (OOB) error as an additional performance measure.
   - Experimented with balancing techniques to address class imbalance in the random forest model.

2. **XGBoost with Multivariate Trees**:
   - Built a model tree with XGBoost, incorporating L1-penalized logistic regression at each node.
   - Applied 5-fold, 10-fold, or leave-one-out cross-validation to estimate error and compare with test error.
   - Evaluated performance with confusion matrix, ROC, and AUC.

## SMOTE for Class Imbalance
1. **SMOTE Pre-processing**:
   - Applied SMOTE (Synthetic Minority Over-sampling Technique) to balance the training data.
2. **Comparison of Results**:
   - Retrained the XGBoost model with SMOTE-preprocessed data.
   - Compared model performance (confusion matrix, ROC, AUC) with the original, unbalanced case.

## Requirements
The project requires:
- Python
- Libraries: `numpy`, `pandas`, `matplotlib`, `sklearn`, `xgboost`, `imblearn`

