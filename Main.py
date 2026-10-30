# The pipeline for the Customer Churn Predictor (same steps as Neural_project.ipynb)
# app.py imports this file, so everything below runs once when the website starts.

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # keep TensorFlow quiet

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # draw plots without a window
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve)
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

# ---------------- 1. Load the data ----------------
df = pd.read_csv("synthetic_customer_churn_100k.csv")

# ---------------- 2. Clean and prepare ----------------
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})        # target to 0/1
X = df.drop(columns=["CustomerID", "Churn"])               # ID has no predictive value
y = df["Churn"]

num_cols = ["Age", "Tenure", "MonthlyCharges", "TotalCharges"]
cat_cols = ["Gender", "Contract", "PaymentMethod"]

preprocess = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
])

# ---------------- 3. Explore (plots for the website) ----------------
sns.set_theme(style="whitegrid")

fig1, ax = plt.subplots(figsize=(7, 4))
sns.countplot(x="Contract", hue="Churn", data=df, ax=ax)
ax.set_title("Churn by contract type (0 = stayed, 1 = left)")

fig2, ax = plt.subplots(figsize=(7, 4))
sns.histplot(data=df, x="Tenure", hue="Churn", bins=30, ax=ax)
ax.set_title("Churn by tenure (months)")

fig3, ax = plt.subplots(figsize=(7, 4))
sns.histplot(data=df, x="MonthlyCharges", hue="Churn", bins=30, ax=ax)
ax.set_title("Churn by monthly charges")

# ---------------- 4. Split and preprocess ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

X_train_p = preprocess.fit_transform(X_train)   # fit on train ONLY
X_test_p = preprocess.transform(X_test)
X_train_p = X_train_p.toarray() if hasattr(X_train_p, "toarray") else X_train_p
X_test_p = X_test_p.toarray() if hasattr(X_test_p, "toarray") else X_test_p

# ---------------- 5. Train the three models ----------------
lr = LogisticRegression(max_iter=1000).fit(X_train_p, y_train)
rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                            random_state=42, n_jobs=-1).fit(X_train_p, y_train)

nn = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_p.shape[1],)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
nn.compile(optimizer=tf.keras.optimizers.Adam(0.001),
           loss="binary_crossentropy", metrics=["accuracy"])
early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                              restore_best_weights=True)
nn.fit(X_train_p, y_train, validation_split=0.1, epochs=100,
       batch_size=256, callbacks=[early_stop], verbose=0)

# ---------------- 6. Compare the models on the test data ----------------
probs = {
    "Logistic Regression": lr.predict_proba(X_test_p)[:, 1],
    "Random Forest": rf.predict_proba(X_test_p)[:, 1],
    "Neural Network": nn.predict(X_test_p, verbose=0).ravel(),
}

rows = []
for name, p in probs.items():
    pred = (p >= 0.5).astype(int)
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, p),
    })
results = pd.DataFrame(rows).round(4)
winner = results.sort_values("ROC-AUC", ascending=False).iloc[0]["Model"]

fig4, ax = plt.subplots(figsize=(6, 5))
for name, p in probs.items():
    fpr, tpr, _ = roc_curve(y_test, p)
    ax.plot(fpr, tpr, label=name)
ax.plot([0, 1], [0, 1], "k--")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("ROC curves")
ax.legend()


# ---------------- 7. Helper used by the website ----------------
def predict_churn(age, gender, tenure, monthly, contract, payment, total):
    """Return {model name: churn probability} for one customer."""
    row = pd.DataFrame([{
        "Age": age, "Gender": gender, "Tenure": tenure, "MonthlyCharges": monthly,
        "Contract": contract, "PaymentMethod": payment, "TotalCharges": total,
    }])
    p = preprocess.transform(row)
    p = p.toarray() if hasattr(p, "toarray") else p
    return {
        "Logistic Regression": float(lr.predict_proba(p)[0, 1]),
        "Random Forest": float(rf.predict_proba(p)[0, 1]),
        "Neural Network": float(nn.predict(p, verbose=0)[0, 0]),
    }
