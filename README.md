# 📉 Customer Churn Predictor

Predict whether a customer will leave (churn) or stay. The prediction uses the customer's age, gender, how long they have been a customer, what they pay, their contract type and their payment method. The project compares Logistic Regression, a Random Forest and a Neural Network (TensorFlow/Keras) and picks the best one.

## Dataset

Synthetic Customer Churn: 100,000 rows, one for every customer, with 9 columns (`synthetic_customer_churn_100k.csv`).

Each row has the target `Churn` (Yes or No), along with age, gender, tenure (1 to 72 months), monthly charges, total charges, contract type and payment method.

About 33% of customers churn and 67% stay, so the classes are imbalanced.

Checks and cleaning steps:

- there are no missing values and no duplicate rows
- `CustomerID` is dropped, because an ID has no predictive value
- `Churn` is converted from No/Yes to 0/1
- numeric columns get a median imputer and are scaled (`StandardScaler`)
- text columns get a most-frequent imputer and are one-hot encoded (`OneHotEncoder`)

## Pipeline (`Neural_project.ipynb`)

1. **Load:** read the CSV with pandas (100,000 rows, 9 columns).
2. **Understand:** `info()`, `describe()`, missing values, duplicates and the churn share.
3. **Explore:** 6 plots (churn count, contract vs churn, payment method vs churn, tenure, monthly charges and age by churn) plus a correlation heatmap.
4. **Prepare:** 80/20 stratified train/test split (80,000 / 20,000 rows). The preprocessing is fitted on the training data only, which gives 14 input features.
5. **Baselines:** Logistic Regression and a Random Forest (200 trees).
6. **Train:** a neural network with early stopping (see below).
7. **Compare:** score all three models on the test set with Accuracy, Precision, Recall, F1 and ROC-AUC. Then draw the confusion matrix and the ROC curves.
8. **Save:** the model and the preprocessing step are saved for later use.

### Neural network

| Layer | Details |
|---|---|
| Input | 14 features |
| Dense | 64 units, ReLU |
| Dropout | 0.3 |
| Dense | 32 units, ReLU |
| Dropout | 0.2 |
| Output | 1 unit, sigmoid |

3,073 trainable parameters. Adam optimizer (learning rate 0.001), binary cross-entropy loss, batch size 256, 10% of the training data used for validation. Early stopping watches the validation loss (patience 8) and restores the best weights.

## Results

Scores on the 20,000 test customers, with a 0.5 threshold:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.7235 | 0.6082 | 0.4660 | 0.5277 | 0.7719 |
| Random Forest | 0.7385 | 0.6301 | 0.5108 | 0.5642 | 0.7932 |
| Neural Network | **0.7614** | **0.6698** | **0.5526** | **0.6056** | **0.8037** |

The Neural Network wins on every metric. It finds about 55% of the customers who really leave, and about 67% of the customers it flags as churners do leave. It is better at spotting customers who stay (recall 0.86) than customers who leave (recall 0.55).

## How to run

Open `Neural_project.ipynb` in Google Colab (or Jupyter) and run the cells from top to bottom. Upload `synthetic_customer_churn_100k.csv` first, so the notebook can find it.

To run it locally:

```
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow joblib
jupyter notebook Neural_project.ipynb
```

The last cell saves two files:

- `churn_model.keras`: the trained neural network
- `preprocess.joblib`: the fitted preprocessing step, which must be applied to new data before predicting

## Project structure

```
churn-predictor/
├── Neural_project.ipynb              # full pipeline: load, explore, prepare, train, compare
├── synthetic_customer_churn_100k.csv # dataset
├── churn_model.keras                 # saved neural network
├── preprocess.joblib                 # saved preprocessing step
└── README.md
```

## Tech used

Python, pandas, NumPy, scikit-learn, TensorFlow/Keras, matplotlib, seaborn, joblib
