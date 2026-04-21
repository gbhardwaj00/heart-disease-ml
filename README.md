# Heart Disease Prediction — Machine Learning Analysis

A machine learning project comparing five classification models for predicting heart disease risk using clinical patient data. Built as part of CMPT 353 at Simon Fraser University.

## Overview

Heart disease is a leading cause of mortality worldwide. This project applies machine learning to a clinical dataset of 918 patients to predict heart disease risk, with the goal of identifying the most effective classification approach for early detection.

## Dataset

- **Source:** Kaggle Heart Disease Dataset
- **Size:** 918 patients, 12 clinical attributes
- **Target:** Binary classification — presence or absence of heart disease

**Features include:** Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope

## Methodology

### Data Preprocessing
- Categorical encoding (binary and ordinal mappings)
- Feature normalization using MinMaxScaler
- Feature selection using SelectKBest (chi-squared test)
- Correlation analysis to identify and drop low-signal features
- Train/test split: 70% training, 30% testing

### Key Features Identified
Based on correlation analysis and SelectKBest scoring, the strongest predictors were ST_Slope, ExerciseAngina, ChestPainType, and Oldpeak.

## Models & Results

| Model | Accuracy | Precision (Heart Disease) | Recall (Heart Disease) |
|---|---|---|---|
| Random Forest | **88%** | 90% | 90% |
| Gradient Boosting | 87% | 90% | 88% |
| MLP Classifier | 87% | 89% | 90% |
| Decision Tree | 86% | 91% | 84% |
| K-Nearest Neighbours | 83% | 90% | 79% |

**Best performer:** Random Forest Classifier with 88% accuracy and balanced precision/recall across both classes.

## Tech Stack

- Python
- pandas, numpy
- scikit-learn (GradientBoostingClassifier, RandomForestClassifier, MLPClassifier, DecisionTreeClassifier, KNeighborsClassifier)
- matplotlib, seaborn

## Project Structure

```
heart-disease-ml/
├── heart-disease-prediction.ipynb   # Main Jupyter notebook
├── features-eng-and-ML-analyze.py   # ML model training and evaluation
├── visualize-and-normalize.py       # EDA and data preprocessing
├── heart.csv                        # Raw dataset
├── normalized_heart_data.csv        # Preprocessed dataset
├── Heart_Attack_Prediction_ML_Model_.pdf  # Full project report
└── *.png                            # Visualization outputs
```

## Key Findings

- **ST_Slope, ExerciseAngina, and ChestPainType** were the strongest predictors of heart disease
- All models showed higher precision for positive (heart disease) cases than negative cases
- Random Forest and Gradient Boosting achieved the best overall balance of precision and recall
- Close training and validation scores across all models indicate well-tuned, generalizable models

## Author

**Gourav Bhardwaj** — Simon Fraser University, CMPT 353
