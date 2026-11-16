# Telco Customer Churn
This telecommunications company provides services such as internet and phone service to its customers, an essential in the era of modern-day technology. However, some customers end up cancelling their subscription, raising some challenges, the most prominent of them being

- Lost revenue: clients provide a monthly/annual subscription fee, losing clients means a lost stream of revenue.
- Customer acquisition costs: replacing the lost customers with new ones may require expensive marketing expenditure
- Customer retention: providing extra, targeted support to customers likely to churn could prevent them from leaving.

The scope of this project is to develop a machine learning classification model capable of identifying customers with a high risk of churning, so that the company can direct retention efforts towards the customers most likely to leave. Model performance is evaluated with metrics that account for both missed churners and unnecessary retention efforts, with the ultimate goal of supporting a more cost-effective retention strategy.

## Dataset
[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM sample data): 7,043 customers, 19 features (demographics, account details, subscribed services), and a binary `Churn` target. About 26.5% of customers churned.

## Approach
All work is in [`churn.ipynb`](churn.ipynb). A 20% test set is held out before any exploration and used only once, at the end.

1. **EDA**: numeric distributions by churn, churn rate by category, contract vs. tenure.
2. **Correlation checks**: Spearman for numeric features, Cramér's V for categorical ones.
3. **Hypothesis tests**:
   - χ² test: contract type vs. churn.
   - Mann–Whitney U: monthly charges of churners vs. stayers.
4. **Metrics**:
   - PR-AUC (primary) and ROC-AUC, for ranking customers by risk.
   - Brier score, to check the probabilities are reliable.
   - Precision, recall and F1 at the chosen threshold.
   - Expected campaign profit, used to choose the threshold.
5. **Feature engineering**: collapse redundant service levels, log-transform tenure, and drop `TotalCharges`, `gender` and `PhoneService`. Each change is checked with repeated cross-validation.
6. **Models**:
   - Baselines: a dummy classifier and Logistic Regression.
   - Candidates, each tuned with a randomised hyperparameter search: Logistic Regression, KNN, Random Forest, HistGradientBoosting and XGBoost.
   - All scored with 5×5 repeated stratified CV.
7. **Post-hoc comparison**: a Friedman test, then a corrected paired t-test (Nadeau–Bengio) of the top model against each other model, with Holm correction.
8. **Final model**: calibration check, profit-based decision threshold, held-out test evaluation.
9. **Interpretation**: SHAP values, cross-checked with permutation importance.

## Results
| | PR-AUC | ROC-AUC | Brier |
|---|---|---|---|
| Logistic Regression (final), 5×5 CV | 0.674 | 0.851 | 0.133 |
| Logistic Regression (final), test set | 0.664 | 0.848 | 0.135 |
| No-skill baseline | 0.265 | 0.500 | 0.195 |

- Logistic Regression ranked first, but its lead over XGBoost, HistGB and Random Forest is not statistically significant; it does significantly beat KNN. It was chosen for its calibration, simplicity and interpretability.
- Log-transforming tenure was the most useful feature engineering step, improving Logistic Regression's PR-AUC by about 0.013.
- At the profit-optimal threshold (0.22), the model contacts 46% of test customers and reaches 84% of the actual churners. The profit figures in the notebook rest on illustrative assumptions (a $50 offer cost, a 30% save rate, 12 months of revenue per saved customer). Swap in real campaign numbers to re-derive the threshold.

## Key drivers of churn
- **Tenure**: risk is highest in the first months and falls steeply until about 20 months.
- **Contract**: month-to-month customers churn about 15× as often as two-year customers.
- **Fibre-optic internet**: these customers have the highest bills and the highest churn.
- **Payment and support**: electronic-check payment and having no online security or tech support are linked to higher churn.

These are associations, not causal effects. A retention lever should be confirmed with an A/B test before it is rolled out.

## Running the notebook
Requires Python 3 with:

```
pip install pandas numpy scipy scikit-learn xgboost shap matplotlib seaborn jupyter
```

Then open `churn.ipynb` and run all cells. It takes about a minute.
