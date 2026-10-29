# YubaIntern
from pathlib import Path

readme = r'''# Customer Churn Prediction and Retention Analysis in the Beauty & Wellness Industry

<p align="center">
  <strong>Data Science with Python Analyst Internship Project</strong><br>
  An end-to-end machine learning project focused on understanding customer churn and supporting data-driven retention strategies.
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white">
  <img alt="Pandas" src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white">
  <img alt="Scikit--learn" src="https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?logo=scikitlearn&logoColor=white">
  <img alt="Project Type" src="https://img.shields.io/badge/Project-Customer%20Analytics-0A7EA4">
</p>

## 📌 Project Overview

Customer retention is an important business objective in the beauty and wellness industry. Losing customers can reduce repeat purchases, revenue, and long-term customer value. This project analyzes historical transaction data, creates customer-level features, builds churn-classification models, and translates model findings into practical retention recommendations.

The project follows a data science workflow: **data understanding → exploratory data analysis (EDA) → feature engineering → predictive modeling → model evaluation → business recommendations**.

## 🎯 Business Problem

The business needs to identify customers who may stop purchasing so that retention teams can prioritize outreach and improve customer engagement.

This project aims to:

- Explore customer purchasing behavior and sales performance.
- Build customer-level features from transaction records.
- Define an operational churn label using a future observation window.
- Train and compare baseline classification models.
- Evaluate performance using metrics suited to an imbalanced target.
- Identify useful churn-related signals and recommend practical retention actions.

## 🗂️ Dataset Summary

The cleaned transaction dataset, `beauty_clean.csv`, contains:

| Item | Value |
|---|---:|
| Transaction records | 51,290 |
| Unique customers | 17,415 |
| Unique orders | 25,728 |
| Date range | 2020-01-01 to 2023-12-31 |
| Missing values in cleaned dataset | 0 |
| Duplicate rows in cleaned dataset | 0 |

The data includes customer, order, product-category, market, sales, quantity, discount, and profit information.

### Churn label definition

For the Week 5 modeling workflow:

- **Historical cutoff:** 2023-06-30
- **Future observation window:** through 2023-09-28 (90 days)
- **Customers in the modeling dataset:** 14,716
- **Operationally churned:** 13,690 (93.03%)
- **Retained:** 1,026 (6.97%)

A customer was labeled **Churned** when they had no purchase during the defined future observation window. This is an operational, time-window-based label; it does **not** prove that a customer has permanently left the business. The strong class imbalance is important when interpreting the results.

## 🧰 Tools & Technologies

- **Python** — analysis and modeling
- **Pandas / NumPy** — data manipulation and numerical analysis
- **Matplotlib / Seaborn** — visual analysis and charts, where used
- **Scikit-learn** — preprocessing, classification, cross-validation, and evaluation
- **Jupyter Notebook** — interactive analysis workflow
- **CSV** — datasets and exported results

> The tools listed reflect the project workflow. Adjust this list to match the exact packages and files in your repository.

## 🔍 Project Workflow

### 1. Data inspection and quality checks

- Loaded the cleaned transaction dataset.
- Reviewed column types, date coverage, customer and order counts.
- Checked for missing values and duplicate records.
- Examined customer segments, product categories, markets, and yearly performance.

### 2. Exploratory Data Analysis (EDA)

EDA was used to understand purchasing patterns and business performance, including:

- Sales, profit, order, and customer trends over time.
- Customer activity, recency, frequency, and monetary behavior.
- Category-level and market-level performance.
- Differences between active and inactive customer groups.

### 3. Feature engineering

Customer-level and transaction-level features were prepared from historical purchase data. Modeling features included:

- `Total_Transactions`
- `Frequency`
- `Monetary`
- `Total_Profit`
- `Total_Quantity`
- `Avg_Discount`
- `Avg_Sales`
- `Unique_Categories`
- `Recency`
- `Average_Order_Value`

`Customer ID` was excluded from model inputs because it is an identifier rather than a behavioral feature. `Unique_Markets` was dropped because it had no variation in the modeling data. Features were created from historical data before the future-window churn label was assigned, helping avoid direct use of future purchases as model inputs.

### 4. Predictive modeling

Two baseline models were trained and compared:

- **Logistic Regression** — used with feature scaling and class weighting.
- **Decision Tree Classifier** — used with class weighting and constraints on tree depth and minimum leaf size.

The dataset was split into training and test sets using an 80/20 stratified split (`random_state=42`).

### 5. Model evaluation

Because the target is highly imbalanced, accuracy alone is not sufficient. The evaluation considered churn-class precision, recall, F1-score, ROC-AUC, PR-AUC, confusion matrices, and stratified five-fold cross-validation.

## 📊 Test Set Results

The following metrics were obtained on the held-out test set:

| Metric | Logistic Regression | Decision Tree |
|---|---:|---:|
| Accuracy | 0.6525 | **0.7038** |
| Precision — Churn | 0.9450 | **0.9486** |
| Recall — Churn | 0.6652 | **0.7207** |
| F1-score — Churn | 0.7808 | **0.8191** |
| ROC-AUC | 0.5863 | **0.6183** |
| PR-AUC | **0.9460** | 0.9496 |

**Best baseline by churn F1-score: Decision Tree.** It achieved an F1-score of **0.8191** and recall of **0.7207** for the Churned class. Its ROC-AUC is still modest, so these results should be treated as a baseline rather than evidence of a production-ready churn system.

### Five-fold cross-validation

| Model | Mean CV F1 | CV F1 Std. Dev. | Mean CV ROC-AUC | Mean CV PR-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.7832 | 0.0140 | 0.5722 | 0.9438 |
| Decision Tree | **0.7842** | 0.0516 | **0.5753** | 0.9408 |

The cross-validation F1 means are close. The Decision Tree has a higher standard deviation, suggesting its F1 performance varied more across folds. More validation and model tuning would be appropriate before deployment.

## 🌳 Decision Tree Feature Importance

The Decision Tree's feature importances were:

| Feature | Importance |
|---|---:|
| `Avg_Discount` | 0.4249 |
| `Frequency` | 0.2000 |
| `Recency` | 0.1016 |
| `Avg_Sales` | 0.0969 |
| `Total_Profit` | 0.0933 |
| `Total_Quantity` | 0.0280 |
| `Monetary` | 0.0226 |
| `Total_Transactions` | 0.0213 |
| `Average_Order_Value` | 0.0115 |
| `Unique_Categories` | 0.0000 |

`Avg_Discount` was the strongest feature according to this model, followed by `Frequency` and `Recency`. Feature importance describes how the trained tree used features; it does not establish causation. The importance values should be checked for stability and potential bias in future experiments.

## 💡 Business Insights

The exploratory analysis and customer comparisons suggest several opportunities:

- **Retention differs by purchasing behavior.** Customers labeled retained had higher average purchase frequency and monetary value than customers labeled churned.
- **Profit contribution matters.** Average total profit was higher for retained customers in the observed comparison.
- **Discount behavior deserves investigation.** Churned customers had a higher average discount in the comparison, and discount was the most important Decision Tree feature. This does not mean discounts cause churn; discount policy and customer mix need further analysis.
- **Category profitability varies.** Home & Accessories showed a negative margin in the analyzed data, while Face Care and Make Up showed stronger margins.
- **Sales growth does not guarantee profit growth.** The analysis showed that 2023 sales increased while year-over-year profit growth declined.

These findings should be interpreted in the context of the dataset, its time period, and the operational churn definition.

## 🚀 Recommendations

1. **Create a retention-priority list.** Use model scores to identify customers for review, then validate the list with business teams before outreach.
2. **Test targeted retention campaigns.** Compare personalized reminders, service messages, loyalty benefits, and carefully controlled offers.
3. **Review discount effectiveness.** Measure incremental repeat purchases and profit—not only redemption or sales volume—before expanding discounts.
4. **Investigate low-margin categories.** Review pricing, sourcing, shipping, returns, and discount practices for categories with weak margins.
5. **Track customer cohorts over time.** Monitor repeat-purchase rate, time to next purchase, churn rate, and customer lifetime value.
6. **Use controlled experiments.** A/B tests or holdout groups can help establish whether retention actions actually improve outcomes.

## 📁 Project Outputs

The Week 4 and Week 5 workflow produced the following analysis outputs:

- `week4_customer_features.csv`
- `week4_transaction_features.csv`
- Business-insights CSV export(s)
- `week5_model_comparison.csv`
- `week5_cross_validation.csv`
- `week5_customer_comparison.csv`
- `week5_class_metrics.csv`
- `week5_project_summary.csv`
- `week5_confusion_matrices.png`
- `week5_model_performance.png`
- `week5_feature_importance.png`

The Week 6 report summarizes evaluation, validation considerations, and business impact recommendations. File availability may vary depending on what has been added to the repository.

## ⚙️ Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook or another Python environment
- The project dataset and exported files, if you want to reproduce the analysis

### Installation

Create and activate a virtual environment if desired, then install the packages used by the notebook:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the common analysis and machine-learning packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### Run the project

1. Clone or download this repository.
2. Place `beauty_clean.csv` in the data directory expected by your notebook or update the dataset path.
3. Open the project's Jupyter Notebook.
4. Run the cells in order: data inspection, EDA, feature engineering, churn-label creation, model training, and evaluation.
5. Review the exported CSV files, charts, and report.

> **Note:** Exact notebook and script filenames were not provided here. Update these instructions to match the files actually committed to your repository. Do not rename or claim outputs that are not included.

## 🧪 Limitations

- The churn label uses a 90-day future observation window and may classify temporarily inactive customers as churned.
- The target is highly imbalanced: only 6.97% of modeled customers were labeled Retained, while 93.03% were labeled Churned.
- The test-set ROC-AUC values are modest, indicating limited ranking/discrimination performance.
- Random stratified splitting may not fully represent future real-world performance. A time-based holdout would provide a stronger deployment simulation.
- Feature importance is not causal evidence.
- Results depend on the available data, its quality, and the defined observation period.
- Model performance and business impact should be re-evaluated on newer data before operational use.

## 🔮 Future Improvements

- Validate with a strictly time-based train/validation/test split.
- Compare additional models such as Random Forest, Gradient Boosting, XGBoost, and LightGBM.
- Tune decision thresholds based on campaign capacity and the cost of false positives/false negatives.
- Evaluate precision-recall curves, calibration, lift, and gains at the top customer segments.
- Test resampling or class-weight strategies carefully, using training data only.
- Add explainability with permutation importance or SHAP, if appropriate.
- Run controlled retention experiments and measure incremental profit.
- Add monitoring for data drift and changes in churn rate.

## ✅ Conclusion

This project demonstrates an end-to-end customer analytics workflow, from transaction-level EDA and feature engineering to churn classification, model evaluation, and business recommendations. The Decision Tree was the strongest of the two tested baselines on held-out churn F1-score, while cross-validation showed that the models' average F1 performance was similar. The results provide a starting point for more rigorous time-based validation, threshold optimization, and retention experiments.

## 🙌 Acknowledgements

Completed as part of the **YuvaIntern Data Science with Python Analyst Internship** project work.

---

<p align="center">
  <i>Built with Python, data analysis, and machine learning.</i>
</p>
'''

path = Path("/mnt/data/README.md")
path.write_text(readme, encoding="utf-8")
print(f"Created: {path}")
print(f"Size: {path.stat().st_size:,} bytes")
