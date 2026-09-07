# Heart Disease Prediction Using Machine Learning

## Overview

This project develops and evaluates a machine learning pipeline for binary heart disease prediction using clinical and demographic patient features.

The project compares conventional machine learning methods with newer pretrained tabular models.

**Final selected model:** TabPFN  
**Selection criterion:** Highest 5-fold cross-validation ROC-AUC  
**Cross-validation ROC-AUC:** 0.9419  
**Final test accuracy:** 89.67%  
**Final test recall:** 92.16%  
**Final test F1-score:** 90.82%  
**Final test ROC-AUC:** 0.9370

> **Important:** This project is for academic and machine-learning evaluation purposes only. It is not a medical diagnostic system and should not replace professional medical judgment.

## Objectives

- Audit and clean a public heart disease dataset.
- Explore relationships between clinical features and the target.
- Build a reproducible binary classification workflow.
- Establish a majority-class baseline.
- Compare conventional and modern tabular classification models.
- Evaluate accuracy, precision, recall, F1-score, and ROC-AUC.
- Analyze false positives and false negatives.
- Interpret important features.
- Document limitations and responsible-use considerations.

## Dataset

**Dataset:** Heart Disease Dataset  
**Source:** Kaggle  
**URL:** https://www.kaggle.com/datasets/eishkaran/heart-disease

The raw dataset contains **1,190 records and 12 columns**. After removing **272 exact duplicate records**, the cleaned dataset contains **918 records**.

### Target

- `0` = No Heart Disease
- `1` = Heart Disease

### Features

| Feature | Type | Description |
|---|---|---|
| `age` | Numerical | Age of the patient |
| `sex` | Categorical | Gender of the patient |
| `chest pain type` | Categorical | Type of chest pain experienced |
| `resting bp s` | Numerical | Resting blood pressure |
| `cholesterol` | Numerical | Serum cholesterol level |
| `fasting blood sugar` | Categorical/Numeric | Fasting blood sugar measurement |
| `resting ecg` | Categorical | Resting electrocardiogram result |
| `max heart rate` | Numerical | Maximum heart rate achieved |
| `exercise angina` | Categorical | Exercise-induced angina |
| `oldpeak` | Numerical | ST depression induced by exercise |
| `ST slope` | Categorical | Slope of the peak exercise ST segment |
| `target` | Binary | Heart disease classification target |

### Data Quality

The audit identified:

- 272 exact duplicate rows, which were removed.
- 172 records with `cholesterol = 0`.
- 1 record with `resting bp s = 0`.

Invalid zero values for resting blood pressure and cholesterol were treated using median-based replacement as described in the project report.

After cleaning:

- Class 0: 410 records (44.69%)
- Class 1: 508 records (55.31%)

## Methodology

### Train/Test Split

An **80/20 stratified split** was used with `random_state=42`.

- Training set: 734 records
- Test set: 184 records

### Preprocessing

Numerical features were standardized using `StandardScaler`.

Categorical features were transformed using `OneHotEncoder` with:

```text
handle_unknown = "ignore"
sparse_output = False
```

The preprocessing produced **22 processed features**.

### Models

Seven approaches were evaluated:

1. Majority-class baseline
2. Logistic Regression
3. Random Forest
4. K-Nearest Neighbors (KNN)
5. TabPFN
6. TabICLv2
7. Mitra

### Validation

Model selection used **5-fold stratified cross-validation ROC-AUC**.

TabPFN achieved the highest cross-validation ROC-AUC (**0.9419**) and was selected as the final model. The final test set was then used for evaluation.

## Results

### Cross-Validation ROC-AUC

| Model | CV ROC-AUC |
|---|---:|
| Majority Baseline | 0.5000 |
| Logistic Regression | 0.9090 |
| Random Forest | 0.9261 |
| KNN | 0.8957 |
| TabPFN | **0.9419** |
| TabICLv2 | 0.9397 |
| Mitra | 0.9340 |

### Final Test Performance

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| KNN | 0.8804 | 0.9000 | 0.8824 | 0.8911 | **0.9385** |
| TabPFN | 0.8967 | 0.8952 | 0.9216 | 0.9082 | 0.9370 |
| TabICLv2 | 0.8967 | 0.8807 | 0.9412 | **0.9100** | 0.9341 |
| Mitra | 0.8913 | 0.8727 | 0.9412 | 0.9057 | 0.9269 |
| Random Forest | 0.8804 | 0.8846 | 0.9020 | 0.8932 | 0.9223 |
| Logistic Regression | 0.8587 | 0.8455 | 0.9118 | 0.8774 | 0.9055 |
| Majority Baseline | 0.5543 | 0.5543 | 1.0000 | 0.7133 | 0.5000 |

TabPFN was selected using the predefined cross-validation criterion, rather than selecting the model based on the final test-set result.

## TabPFN Error Analysis

The selected TabPFN model produced:

| | Predicted No Disease | Predicted Disease |
|---|---:|---:|
| **Actual No Disease** | 71 | 11 |
| **Actual Disease** | 8 | 94 |

- True Negatives: 71
- False Positives: 11
- False Negatives: 8
- True Positives: 94
- Total errors: 19
- Error rate: 10.33%

False negatives are particularly important in this healthcare-related classification task because they represent cases with heart disease classified as not having heart disease.

## Visualizations and Outputs

The notebook generates:

- Target class distribution
- Numerical feature distributions
- Feature correlation heatmap
- Feature-versus-target plots
- Confusion matrices
- ROC curves
- Precision-Recall curves
- Model metric comparison charts
- Feature-importance results
- TabPFN error-analysis results

When the figure-saving cells are enabled, figures are stored in:

```text
/content/figures/
```

when running in Google Colab.

The notebook also generates:

```text
heart_disease_model_comparison.csv
heart_disease_cv_results.csv
heart_disease_final_test_results.csv
heart_disease_tabpfn_errors.csv
```

## Project Structure

A recommended repository structure is:

```text
Heart Disease Prediction Using Machine Learning/
│
├── data/
│   └── heart_statlog_cleveland_hungary_final.csv
│
├── outputs/
│   ├── Chest Pain Type vs Target.png
│   ├── Exercise-Induced Angina vs Target.png
│   ├── Feature Correlation Heatmap.png
│   ├── .....
│
├── src/
│   └── Python_Final.ipynb
│
├── .gitattributes
├── Heart Disease Prediction Using Machine Learning...
└── README.md
```

The exact filenames may vary depending on the submitted notebook version.

## Requirements

The project was developed in **Google Colab** using Python.

Recorded environment versions:

```text
Python        3.13.15
NumPy         2.1.3
Pandas        2.2.3
Scikit-learn  1.6.1
Matplotlib    3.10.0
Seaborn       0.13.2
```

The project also requires packages for:

- TabPFN
- TabICLv2 / TabICL
- Mitra

For TabPFN, authentication may be required when the model is initialized.

## How to Run

### Google Colab

1. Open the submitted `.ipynb` notebook in Google Colab.
2. Upload the permitted dataset CSV.
3. Install any missing dependencies.
4. If TabPFN is missing, for example:

```python
!pip install -q tabpfn
```

5. Restart the runtime if Colab requests it.
6. Run the notebook **sequentially from the first cell to the final evaluation section**.
7. Provide the required TabPFN authentication key if prompted.
8. Review the generated tables, metrics, figures, and error-analysis outputs.

Later notebook sections depend on objects created earlier, so running the notebook from beginning to end is recommended.

## Reproducibility

The project uses:

```text
random_state = 42
```

The train/test split is stratified, and model evaluation uses stratified 5-fold cross-validation with shuffling and a fixed random state.

## Limitations

- The dataset is relatively small compared with real-world clinical populations.
- It may not represent all patient populations or healthcare settings.
- Duplicate records and invalid values were present in the original data.
- Performance may differ on independent or external datasets.
- The current preprocessing implementation has a methodological limitation: the median values used to replace invalid resting blood pressure and cholesterol values were calculated before the train/test split. Therefore, the imputation statistics were not learned strictly from the training partition.
- The results should not be interpreted as evidence of clinical effectiveness.

## Responsible Use

This project is intended for **academic machine-learning evaluation only**.

Predictions should **not** be used to diagnose heart disease or make medical decisions. Real-world use would require independent clinical validation, assessment across relevant patient groups, and professional clinical oversight.
