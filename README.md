# Reproducing "Empirical Comparison of Supervised Learning Algorithms" (2025 Edition)

**Author:** Wenyao Yu  
**Affiliation:** University of California, San Diego  
**Email:** w6yu@ucsd.edu  

## 📌 Abstract
This project reproduces and extends the seminal empirical study by Caruana and Niculescu-Mizil (ICML 2006) on supervised learning algorithms. Using modern Python tools available in 2025, we evaluate five classifier families across four UCI datasets. 

The study validates that tree-based ensembles (XGBoost, Random Forests) continue to dominate on noisy, real-world data, while SVMs with RBF kernels excel on clean, low-dimensional tasks. Modern implementations yielded superior performance benchmarks compared to the original 2006 results (e.g., **86.77%** on Adult vs ~85.7%).

## 🚀 Key Features
* **Modern Stack:** Replaced legacy C++/Java implementations with `scikit-learn` and `XGBoost`.
* **Expanded Scale:** Added the **Cod-RNA** dataset (100k instances) to test high-dimensional performance.
* **Reproducibility:** Implemented a robust pipeline with **checkpointing** to resume long-running experiments.
* **Rigorous Tuning:** Utilized stratified 3-fold/5-fold cross-validation with `GridSearchCV` and `RandomizedSearchCV`.

## 📊 Datasets
Four datasets were utilized, retrieved via `ucimlrepo` or loaded from `svmlight` formats.

| Dataset | Instances | Features | Type | Task |
| :--- | :--- | :--- | :--- | :--- |
| **Adult** | 48,842 | 14 | Mixed | Income >$50K |
| **Spambase** | 4,601 | 57 | Numeric | Spam vs Non-Spam |
| **Letter** | 20,000 | 16 | Numeric | Letter 'A' vs Rest |
| **Cod-RNA*** | 100,000 | 8 | Numeric | Coding vs Non-Coding |

*\*Cod-RNA was subsampled from 331,152 instances for computational efficiency while retaining scale.*

## 🧠 Methods & Models
We evaluated five classifier families using 2025 industry-standard libraries:

1.  **XGBoost:** A scalable tree boosting system (replacing the original Boosted Trees).
2.  **Random Forest:** Standard `RandomForestClassifier`.
3.  **SVM-RBF:** `SVC` with Probability Calibration.
4.  **Neural Net (MLP):** `MLPClassifier` with early stopping.
5.  **KNN:** `KNeighborsClassifier` (k-Nearest Neighbors).

### Preprocessing Pipeline
* **Numeric:** Standardized using `StandardScaler`.
* **Categorical:** One-hot encoded via `OneHotEncoder` (handling unknown categories).
* **Pipeline:** Managed via `ColumnTransformer` to handle mixed-type datasets like *Adult* correctly.

## 📈 Results Summary
The table below shows the accuracy on the **80% training split** (averaged over 3 runs).

| Model | Adult | Spambase | Letter | Cod-RNA |
| :--- | :--- | :--- | :--- | :--- |
| **SVM-RBF** | 0.8562 | 0.9337 | **0.9995** | 0.9554 |
| **Random Forest** | 0.8565 | 0.9533 | 0.9972 | 0.9591 |
| **XGBoost** | **0.8658** | **0.9540** | 0.9980 | **0.9652** |
| **Neural Net** | 0.8577 | 0.9334 | 0.9973 | 0.9604 |
| **KNN** | 0.8445 | 0.9175 | 0.9980 | 0.9433 |

**Key Findings:**
* **XGBoost** won 3 out of 4 datasets, proving its dominance on mixed and noisy data.
* **SVM-RBF** achieved near-perfect accuracy on the *Letter* dataset, consistent with 2006 findings on clean data.
* Accuracy consistently improved with larger training fractions (20% -> 50% -> 80%).
