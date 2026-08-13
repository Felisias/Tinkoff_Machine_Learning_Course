#  Tinkoff Education: Machine Learning Coursework

![Status](https://img.shields.io/badge/Status-In_Progress-orange.svg)
![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-Mathematics-lightblue.svg)

##  About This Repository
This repository contains a comprehensive collection of practical tasks, algorithmic implementations, and data processing scripts completed as part of the **Machine Learning** course at **T Education** (Currently in progress). 

The focus of these tasks is on implementing core machine learning algorithms and metrics "from scratch" using `numpy`, which provides a deep mathematical understanding of how popular libraries like `scikit-learn` operate under the hood.

##  Repository Structure

The repository consists of n individual task files covering various domains of Machine Learning, Data Preprocessing, and Deep Learning components. 

###  1. Linear Models & Regressions
* `SGDPoissonRegression` (with Momentum)
* `SGDLinearRegression` & `SGDLogisticRegression`
* `RidgeRegression` & `RidgeCV`
* `LassoRegression` (Coordinate Descent)
* `Perceptron`

###  2. Tree-Based Models & Ensembles
* Impurity metrics: `gini_impurity`, `entropy`
* Ensemble parameters: `calculate_adaboost_alpha`

###  3. Clustering & Distance-Based Models
* `KMeans` & `DBSCAN`
* `KNNClassifier` & `KNNRegressor`
* Distance Metrics: `euclidean_distance`, `manhattan_distance`, `minkowski_distance`, `cosine_similarity`

###  4. Data Preprocessing & Feature Engineering
* **Scaling:** `StandardScaler`, `MinMaxScaler`, `Normalizer` (L2)
* **Encoding:** `LabelEncoder`, `OneHotEncoder`, `Binarizer`
* **Imputation:** `SimpleImputer` (Mean)
* **Feature Extraction:** `PolynomialFeatures`, `PCA`
* **NLP:** `term_frequency` (TF), `inverse_document_frequency` (IDF)
* **Feature Selection:** Variance Threshold, L1 features, P-value OLS, Mutual Info, Chi-Square, Pearson correlation, Permutation Importance.

###  5. Metrics & Validation
* **Classification:** `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`, `confusion_matrix`, `jaccard_score`
* **Regression:** `mean_squared_error` (MSE), `mean_absolute_error` (MAE), `r2_score`
* **Loss Functions:** `log_loss` (Cross-Entropy), `huber_loss`, `hinge_loss`
* **Validation:** `train_test_split`, `KFold` cross-validation, `bootstrap_sample`

###  6. Deep Learning & Optimization Components
* **Optimizers:** `gradient_descent`, `AdamOptimizer`, `RMSpropOptimizer`
* **Learning Rate:** `exponential_decay`
* **Activations & Gradients:** `relu`, `sigmoid`, `softmax`, and their derivatives.
* **Layers & Regularization:** `dropout_forward`, `batch_norm_forward`, `max_pooling_1d`, L1/L2 gradients.
* **Kernels:** `rbf_kernel`, `polynomial_kernel`
