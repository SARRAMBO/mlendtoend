# End-to-End Machine Learning Regression Pipeline

This repository contains a comprehensive, production-grade implementation of an end-to-end Machine Learning regression pipeline. It tracks performance shifts from a baseline linear estimator up through higher-degree polynomial feature architectures.

## 🛠️ Project Architecture & Pipeline

1. **Exploratory Data Analysis (EDA):** Pre-processed structural nulls and isolated target metrics from a dataset containing `13,828` samples.
2. **Feature Scaling:** Applied `StandardScaler` variance scaling to prevent magnitude bias across variables.
3. **Polynomial Feature Engineering:** Explored interactions by projecting the 8-feature base dimension across higher dimensions:
   * **Degree 2:** Generates 45 interactive features.
   * **Degree 3:** Generates 165 interactive features.
   * **Degree 4:** Generates 495 interactive features.
4. **Estimator Evaluation:** Trained standard Ordinary Least Squares (OLS) `LinearRegression` estimators against every degree block.

---

## 📊 Performance Insights & The Bias-Variance Tradeoff

| Model Feature Space | Complexity (Features) | Performance Behavior | Valuation |
| :--- | :---: | :--- | :--- |
| **Linear Baseline (Degree 1)** | 8 | High Bias (Underfits slightly) | Stable baseline |
| **Polynomial (Degree 2)** | 45 | **Optimal Balance** | **Best generalization curve** |
| **High Polynomials (Degree 3 & 4)** | 165 / 495 | High Variance (Overfitting) | Model collapses on test space |

### Overfitting Case Study
During testing, advancing the network complexity to **Degree 4 (495 dimensions)** caused standard OLS calculations to overfit training noise entirely. While training error diminished, the **Adjusted $R^2$ on the evaluation test subset plummeted into negative margins**, visualizing the classic Machine Learning pitfall of high model variance.

---

## 💾 Model Production Architecture (Pickling)

To guarantee deployment reproducibility without retraining overhead, the optimum pipeline assets were serialized into persistent byte streams using Python's Native `pickle` module:

* `poly_transformer.pkl` — Captures the extracted degree weights and feature maps.
* `trained_linear_regressor.pkl` — Stores the final optimized OLS coefficients.

### Deployment Quickstart
```python
import pickle

# Load production components
with open("poly_transformer.pkl", "rb") as f:
    poly_transformer = pickle.load(f)
with open("trained_linear_regressor.pkl", "rb") as f:
    model = pickle.load(f)

# Input new production records -> scale -> predict
# target_predictions = model.predict(poly_transformer.transform(new_raw_features))
