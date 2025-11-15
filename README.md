# Fairness and Explainability in Income Classification

This project builds, evaluates, and interprets a Logistic Regression model on the Adult Income dataset, with a focus on **fairness** and **explainable AI**.

---

## 🎯 Objectives

- Load and preprocess a public dataset suitable for fairness analysis  
- Train a binary classifier (Logistic Regression) to predict income (`<=50K` vs `>50K`)  
- Evaluate model performance with traditional metrics  
- Assess fairness across a sensitive attribute (**sex**) using Fairlearn  
- Explain model behaviour using **SHAP** (global & local) and **LIME** (local)

---

## 📂 Dataset

- **Name:** Adult Income (UCI Machine Learning Repository)  
- **Source URL:** `https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data`  
- **Target variable:** `income` (`<=50K`, `>50K`)  
- **Sensitive feature:** `sex` (Female, Male)  

Key features used include:
- Demographics: `age`, `race`, `sex`, `marital-status`  
- Socioeconomic: `education`, `education-num`, `workclass`, `occupation`, `hours-per-week`  
- Financial: `capital-gain`, `capital-loss`  

Categorical features are one-hot encoded; numerical features are standardized.

---

## 🧠 Model

- **Algorithm:** Logistic Regression (`sklearn.linear_model.LogisticRegression`)  
- **Train/test split:** 70% train, 30% test (`train_test_split`, `stratify=y`)  
- **Scaling:** `StandardScaler` applied to all features after one-hot encoding  

### Performance

On the test set:

- **Accuracy:** ~0.854  
- **Class 0 (<=50K)**  
  - Precision: 0.88  
  - Recall: 0.93  
  - F1-score: 0.91  
- **Class 1 (>50K)**  
  - Precision: 0.74  
  - Recall: 0.61  
  - F1-score: 0.67  

Confusion Matrix:

\[
\begin{bmatrix}
6902 & 515 \\
914  & 1438
\end{bmatrix}
\]

The model performs better on the majority class (`<=50K`) than on the minority class (`>50K`).

---

## ⚖️ Fairness Analysis

We use **Fairlearn** to evaluate fairness across the sensitive attribute **sex** (`0 = Female`, `1 = Male`), using:

- `selection_rate`
- `false_positive_rate`
- `true_positive_rate`

Example MetricFrame results on the test set:

| sex | selection_rate | false_positive_rate | true_positive_rate |
|-----|----------------|---------------------|--------------------|
| 0 (Female) | ~0.085 | ~0.025 | ~0.559 |
| 1 (Male)   | ~0.256 | ~0.097 | ~0.621 |

Interpretation:

- Males are **more than 3×** as likely to be predicted as `>50K` compared to females (higher selection rate).  
- Males also have a higher false positive rate and slightly higher true positive rate.  
- This reveals **disparate impact** and potential bias in favour of male predictions.

A bar chart is used to visualize these metrics by group.

---

## 🔍 Explainability

### SHAP (Global & Local)

We use **`shap.LinearExplainer`** on the trained Logistic Regression model.

- **Global (summary bar plot)** shows the most influential features overall, including:  
  - `marital-status_Married-civ-spouse`  
  - `capital-gain`  
  - `education-num`  
  - `sex_Male`  
  - `age`  
  - `workclass_Private`  

- **Global (beeswarm plot)** reveals how high vs low feature values push predictions towards `>50K` or `<=50K`.

- **Local (waterfall plot)** explains individual decisions by showing how each feature increases or decreases the log-odds of predicting `>50K` for a single person.

### LIME (Local Explanations)

We use **`LimeTabularExplainer`**:

- Creates a locally linear surrogate model around a single instance.  
- Returns feature contributions that push the prediction towards `<=50K` or `>50K` with interpretable rules, e.g.:
  - `capital-gain <= threshold` → supports `<=50K`  
  - `marital-status_Married-civ-spouse` and certain occupations → support `>50K`  


```bash
pip install fairlearn shap lime
