# Responsible AI: Fairness & Explainability in Income Classification

> **Looking for my full portfolio?** [Start here: Business Analysis, Automation & Data Analytics Portfolio](PORTFOLIO.md)
>
> CBAP · Salesforce Certified Business Analyst · Workflow Analysis · Simplification, Automation & Data Integrity

**Adetola Ridwat Odulaja** · Academic responsible AI project

Evaluate an income classifier beyond aggregate accuracy by examining group-level outcomes and explaining individual predictions.

**Focus:** Responsible AI · Model evaluation · Fairness analysis · Explainability

## Question

Does a model with good overall accuracy behave similarly across groups, and which features influence its decisions?

## Approach

The notebook uses the UCI Adult Income dataset, one-hot encoding, a stratified 70/30 train/test split, and scaling fitted on training data. Logistic Regression predicts whether recorded income exceeds $50,000. Fairlearn compares selection, false-positive, and true-positive rates by recorded sex. SHAP and LIME explain model behavior globally and locally.

## Recorded results

The saved run reports **85.37% accuracy** on 9,769 test records. Recall for the higher-income class is approximately 61%.

| Recorded group | Selection rate | False-positive rate | True-positive rate |
|---|---:|---:|---:|
| Female | 8.47% | 2.50% | 55.87% |
| Male | 25.60% | 9.70% | 62.09% |

These differences warrant investigation. They do not, by themselves, establish the cause of disparities or determine whether a model is fair for a particular use.

## Why it matters

An overall score can conceal unequal error patterns. This project demonstrates the value of combining performance metrics, group comparisons, and explanations when documenting AI risks and communicating findings to stakeholders.

## Explore the work

- [Executed notebook](Ethical_AI_Analysis_and_Explainability.ipynb)
- [Submitted report](Ethical%20AI%20Analysis.pdf)
- [Full portfolio](PORTFOLIO.md)

## Run

Open the notebook in Google Colab and run sequentially. It downloads Adult data from the URL embedded in the notebook and installs Fairlearn, SHAP, and LIME. Other dependencies include Pandas, NumPy, scikit-learn, and Matplotlib.

## Limitations

This is an academic analysis of historical data. Sensitive attributes remain among the model inputs. Fairness is examined along one attribute, and explanation tools do not establish causality or fairness. Missing-value parsing should be checked: the saved sample counts retain all 32,561 source records despite a drop-missing step. No fairness mitigation, approval for operational use, or production deployment is established.

Results above were checked against saved notebook outputs; the analysis was not rerun in this documentation update.
