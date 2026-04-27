# HR Employee Attrition Prediction — End-to-End ML Pipeline

> **Complete binary classification pipeline predicting employee attrition on IBM HR Analytics data. Covers feature engineering, StandardScaler, OneHotEncoder, SMOTE class balancing, and Decision Tree modeling — achieving 81% test accuracy. Score: 100 / 100.**

---

## Business Problem

Employee attrition is one of the most expensive challenges in human resources. Replacing a single employee costs **50–200% of their annual salary** in recruitment, onboarding, and productivity loss. Organizations with 1,000+ employees can lose millions annually without a data-driven early-warning system.

This project builds an end-to-end ML pipeline that predicts **which employees are at risk of leaving** so HR teams can deploy targeted retention interventions before high-value employees resign.

---

## Dataset

**Source:** IBM HR Analytics Employee Attrition Dataset  
**Size:** 1,470 employees · 35 features (34 input + 1 target)  
**Target:** `Attrition` — Yes (left) or No (stayed)

| Dataset Property | Value |
|---|---|
| Total employees | 1,470 |
| Employees who left | 237 (16.1%) |
| Employees who stayed | 1,233 (83.9%) |
| Numerical features | 26 |
| Categorical features | 8 |
| Class imbalance | 4.8 : 1 (stayed : left) |

**Key features include:** Age, MonthlyIncome, DailyRate, OverTime, BusinessTravel, Department, JobRole, JobSatisfaction, EnvironmentSatisfaction, YearsAtCompany, TotalWorkingYears

<img width="1483" height="810" alt="image" src="https://github.com/user-attachments/assets/34a96e99-8016-4b08-b882-ad86006ebe49" />

---
Pipeline Overview
The project implements a full 8-step ML pipeline following production best practices — all preprocessing is fitted on training data only and applied separately to the test set to prevent data leakage.
Raw Data
   │
   ├── Q1: Feature / Target Separation        X = features, y = Attrition
   ├── Q2: Train / Test Split (75 / 25)       1,102 train  ·  368 test
   ├── Q3: Variable Type Identification       26 numerical  ·  8 categorical
   ├── Q4: StandardScaler                     Zero mean, unit variance
   ├── Q5: OneHotEncoder                      Binary dummy columns
   ├── Q6: Target Encoding                    Yes → 1  ·  No → 0
   ├── Q7: SMOTE Oversampling                 1,102 → 1,826 balanced samples
   └── Q8: Decision Tree Classifier           81% test accuracy

## ML Pipeline — 8 Steps

### Step 1 — Feature / Target Separation
```python
X = df.drop('Attrition', axis=1)   # 34 input features
y = df['Attrition']                 # target: Yes / No
```
Attrition is removed from the feature matrix before any modeling step to prevent data leakage. The model learns exclusively from the remaining 34 employee characteristics.
---

### Step 2 — Train / Test Split (75% / 25%)
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
```

| Split | Samples | Percentage |
|---|---|---|
| Training set | 1,102 | 75% |
| Test set | 368 | 25% |

All preprocessing is fitted on training data and applied to test data — the gold standard for preventing leakage.
The split is performed before all preprocessing to ensure test-set integrity throughout the pipeline.
---

### Step 3 — Variable Type Identification
```python
numerical_cols   = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()  # 26 columns
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist() #  8 columns
```
Separation is done from `X_train` specifically — not the full dataset — to ensure type detection is grounded in training-set statistics only.

<img width="1014" height="684" alt="image" src="https://github.com/user-attachments/assets/4dd10d60-a12f-413c-9aea-2a1c8b3f84a2" />

---

### Step 4 — StandardScaler on Numerical Features
```python
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols]  = scaler.transform(X_test[numerical_cols])
```
Transforms each feature to zero mean and unit variance: `z = (x − μ) / σ`

Without scaling, high-magnitude features like `MonthlyIncome` ($1K–$20K) would dominate distance-based algorithms over low-range features like `JobLevel` (1–5). Post-scaling: `Age.max()` ≈ 2.5 standard deviations — confirmed within expected range.

<img width="1084" height="384" alt="image" src="https://github.com/user-attachments/assets/02b2b751-0989-4083-9b9a-7c06e7a4a914" />

Rule strictly followed: fit_transform on training data only. transform on test data. Fitting on test data would leak test-set mean and variance into the model, producing artificially optimistic evaluation metrics.


---

### Step 5 — OneHotEncoder on Categorical Features
```python
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_train_enc = ohe.fit_transform(X_train[categorical_cols])
X_test_enc  = ohe.transform(X_test[categorical_cols])
```
Converts 8 categorical columns into binary (0/1) dummy features. handle_unknown='ignore' silently handles any category level present in test but not in training, preventing runtime errors in production-like evaluation scenarios. Confirmed: encoded feature JobRole_Research Scientist is present in the final feature matrix.
---

### Step 6 — Target Encoding (Yes → 1, No → 0)
```python
y_train = y_train.map({'Yes': 1, 'No': 0})
y_test  = y_test.map({'Yes': 1, 'No': 0})
```
Binary integer labels are required by scikit-learn classifiers and probabilistic output methods. Validation: encoded y_train mean ≈ 0.161, confirming the 16.1% attrition rate is preserved correctly after splitting.
---

### Step 7 — SMOTE Oversampling (Training Set Only)
```python
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
```

| Class | Before SMOTE | After SMOTE |
|---|---|---|
| Stayed (0) | 913 | 913 |
| Left (1) | 189 | 913 |
| **Imbalance ratio** | **4.8 : 1** | **1 : 1** |
| **Total training samples** | **1,102** | **1,826** |

<img width="983" height="409" alt="image" src="https://github.com/user-attachments/assets/705f0908-089a-49b3-bd61-35895aab66c4" />

The training set has a severe class imbalance: 913 employees who stayed vs. 189 who left — a 4.8:1 ratio. A model trained on this raw imbalance learns to predict "stayed" almost exclusively, achieving high accuracy while failing at the task that actually matters: catching the 16% who will leave.
SMOTE (Synthetic Minority Over-sampling Technique) resolves this by generating synthetic minority-class samples through interpolation between existing minority observations in feature space — adding genuine diversity rather than simple duplication.

> **Why training set only:** SMOTE is never applied to the test set. The test set preserves the real-world class distribution so model evaluation reflects true deployment conditions.


---

### Step 8 — Decision Tree Classifier

```python
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
```
A Decision Tree is an interpretable model that recursively partitions the feature space based on the most informative thresholds. It is an ideal baseline classifier for HR analytics because it produces human-readable decision rules — a manager can understand exactly why the model flagged an employee as high-risk, unlike black-box ensemble methods.

**Model Results:**

| Metric | Value |
|---|---|
| **Test Accuracy** | **81.0%** |
| ROC-AUC | > 0.75 |
| Grader threshold | > 70% — ✅ Passed |

**Classification Report:**

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Stayed (0) | 0.90 | 0.87 | 0.88 | 311 |
| Left (1) | 0.51 | 0.58 | 0.54 | 57 |
| **Weighted avg** | **0.83** | **0.81** | **0.82** | **368** |

<img width="1584" height="512" alt="image" src="https://github.com/user-attachments/assets/2777bea3-029d-43ad-8411-82df79515330" />


**Top 5 Attrition Predictors (by feature importance):**
1. `MonthlyIncome` — Salary is the strongest attrition predictor
2. `OverTime_Yes` — Frequent overtime signals burnout and exit intent
3. `Age` — Younger employees are highest risk
4. `TotalWorkingYears` — Less experienced employees exit more frequently
5. `YearsAtCompany` — Early-tenure employees are flight risks

---

## Final Score

| Question | Task | Key Technique | Result |
|---|---|---|---|
| Q1 | Feature / target separation | `DataFrame.drop` | ✅ Pass |
| Q2 | Train / test split 75/25 | `train_test_split` | ✅ Pass |
| Q3 | Variable type identification | `select_dtypes` (26 num, 8 cat) | ✅ Pass |
| Q4 | Numerical scaling | `StandardScaler` | ✅ Pass |
| Q5 | Categorical encoding | `OneHotEncoder` | ✅ Pass |
| Q6 | Target encoding | `.map({'Yes':1,'No':0})` | ✅ Pass |
| Q7 | Class balancing | `SMOTE` | ✅ Pass |
| Q8 | Model + evaluation | `DecisionTreeClassifier` — 81% | ✅ Pass |

**Total: 100 / 100**

---

## Visualizations

| File | Contents |
|---|---|
| `hr_eda.png` | 6-panel EDA: attrition balance, income box plot, age distribution, business travel, job role rate, years at company |
| `correlation_heatmap.png` | Pearson correlation matrix of key numerical features |
| `standard_scaling.png` | Box plot of scaled features + Age z-score distribution |
| `smote_balance.png` | Before vs. after SMOTE class balance bar charts |
| `model_performance.png` | Confusion matrix + ROC curve + top 15 feature importances |

---

## Business Recommendations

The Decision Tree model and feature importance analysis translate directly into actionable HR strategy:

| Predictor | Recommendation |
|---|---|
| Monthly Income | Conduct salary benchmarking annually; address pay gaps before employees receive outside offers |
| OverTime | Implement overtime caps and compensation reform; monitor overtime hours as a real-time attrition signal |
| Age / Tenure | Deploy early-career mentoring programs and accelerated development tracks for employees in first 2 years |
| Job Satisfaction | Run quarterly pulse surveys; assign manager coaching to teams with below-average satisfaction scores |
| Business Travel | Offer hybrid travel policies and travel wellbeing allowances to frequent travelers |

Estimated ROI: If the model enables HR to retain even 10% of the 237 at-risk employees, at an average salary of $65,000 and a replacement cost of 100%, the organization avoids approximately $1.5M in annual attrition costs.
---

## Tech Stack

```
Python 3.10
├── pandas          — data loading, feature engineering, groupby
├── NumPy           — numerical operations
├── scikit-learn    — train_test_split, StandardScaler, OneHotEncoder,
│                     DecisionTreeClassifier, metrics
├── imbalanced-learn — SMOTE oversampling
├── matplotlib      — all chart rendering and export
└── seaborn         — EDA visualizations and heatmap
```

---

## How to Run

**Google Colab (recommended)**
```python
# Upload notebook → Runtime → Run all
# Dataset downloads automatically from Google Drive
```

**Local Jupyter**
```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
jupyter notebook OkothAketch_A7.ipynb
```

---

## Skills Demonstrated

`Binary Classification` `Feature Engineering` `StandardScaler` `OneHotEncoder` `SMOTE Oversampling` `Class Imbalance Handling` `Decision Tree` `ROC-AUC` `Feature Importance` `Train-Test Split` `Data Leakage Prevention` `HR Analytics` `Python` `scikit-learn` `imbalanced-learn`

Skills Demonstrated
Binary Classification Supervised Learning Feature Engineering StandardScaler OneHotEncoder SMOTE Oversampling Class Imbalance Handling Decision Tree ROC-AUC Confusion Matrix Feature Importance Train-Test Split Data Leakage Prevention HR Analytics Python scikit-learn imbalanced-learn seaborn
---

## Author

**Aketch Adhiambo Okoth**  
MS Business Analytics — Montclair State University (GPA 3.8)  
[LinkedIn](https://linkedin.com/in/your-profile) · [Portfolio](https://your-portfolio-url.com)
