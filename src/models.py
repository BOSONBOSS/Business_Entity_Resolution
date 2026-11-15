"""
Machine Learning Models Factory (CSE2032 Syllabus Alignment)
------------------------------------------------------------
Implements the 6 core machine learning paradigms required by the course syllabus:
  1. Linear / Regularized: Logistic Regression (L2 Ridge & L1 Lasso)
  2. Instance-Based: K-Nearest Neighbors (KNN with distance weighting)
  3. Margin-Based: Support Vector Machine (RBF Kernel & Linear Kernel)
  4. Tree-Based: Decision Tree (Pruned depth=8 with Gini Impurity)
  5. Bagging Ensemble: Random Forest Classifier (100 estimators)
  6. Boosting Ensemble: LightGBM Gradient Boosted Trees
"""

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

def get_models(random_state=42):
    """
    Returns an ordered dictionary of model names and configured estimator instances.
    """
    models = {
        'Logistic_Regression_L2': LogisticRegression(
            penalty='l2',
            C=1.0,
            solver='lbfgs',
            max_iter=1000,
            random_state=random_state
        ),
        'Logistic_Regression_L1': LogisticRegression(
            penalty='l1',
            C=1.0,
            solver='saga',
            max_iter=1000,
            random_state=random_state
        ),
        'K_Nearest_Neighbors': KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            metric='euclidean',
            n_jobs=-1
        ),
        'Support_Vector_Machine_RBF': SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            max_iter=3000,
            random_state=random_state
        ),
        'Decision_Tree_Pruned': DecisionTreeClassifier(
            max_depth=8,
            criterion='gini',
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=random_state
        ),
        'Random_Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=10,
            random_state=random_state,
            n_jobs=-1
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=100,
            learning_rate=0.08,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            verbose=-1,
            n_jobs=-1
        )
    }
    return models

