import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, matthews_corrcoef, roc_auc_score, 
                             mean_absolute_error, mean_squared_error, r2_score)

class MLTrainEngine: # <--- Make sure this name is EXACTLY this
    def __init__(self):
        # The 8 core algorithms
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=10000),
            "Random Forest": RandomForestClassifier(n_estimators=100),
            "SVM": SVC(probability=True),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            "Decision Tree": DecisionTreeClassifier(),
            "KNN": KNeighborsClassifier(),
            "AdaBoost": AdaBoostClassifier(),
            "Gradient Boosting": GradientBoostingClassifier()
        }
        self.trained_models = {}

    def run_all(self, X_train, X_test, y_train, y_test):
        performance = []
        for name, model in self.models.items():
            # Training
            model.fit(X_train, y_train)
            self.trained_models[name] = model
            
            # Predicting
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds
            
            # Metrics
            acc = accuracy_score(y_test, preds)
            mcc = matthews_corrcoef(y_test, preds)
            auc = roc_auc_score(y_test, probs)
            
            performance.append({
                "Model": name,
                "Accuracy": acc,
                "MCC": mcc,
                "AUC": auc
            })
            
        return pd.DataFrame(performance)