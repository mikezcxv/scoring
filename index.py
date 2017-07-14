import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, auc, roc_auc_score, roc_curve, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, normalize, MinMaxScaler
from sklearn.grid_search import GridSearchCV

dir_path = os.path.dirname(os.path.realpath(__file__))

df = pd.read_csv(dir_path + '/Train.csv')
df_test = pd.read_csv(dir_path + '/Test_origin.csv')
df_test_original = df_test.copy()

# Let's take a look at the data
# print(df.info())

# We have 15 independent vars; Object: 1; Numerical: 14
# "Seniority", "Time", "Age", "Marital", "Records",
# "Expenses", "Income", "Assets", "Debt", "Amount",
# "Price", "ResidenceType", "EmploymentType", "ApplicationId", "PhoneNumber"

# 5 vars have missing values:
# Income            3267 non-null float64
# Assets            3527 non-null float64
# Debt              3549 non-null float64
# ResidenceType     3558 non-null float64
# EmploymentType    3562 non-null float64

# #print(df['PhoneNumber'].value_counts())
# PhoneNumber seems to be a noise
# We have duplicate for phone 144057176170
# print(df[df['PhoneNumber'] == 144057176170])
df.drop([3322], inplace=True)

# There is one missing value in a target variable
df_skip = df.loc[pd.isnull(df['Status'])]
df = df.loc[~pd.isnull(df['Status'])]

df['Status'] = df['Status'].map({'bad': 0, 'good': 1})
# 1    2546 - good
# 0    1017 - bad


def format_data(raw):
    raw['Records'] = raw['Records'].map({'no_rec': 0, 'yes_rec': 1})

    # Set the most frequent 'Marital' for one missing record in test data
    raw.loc[pd.isnull(raw['Marital']), 'Marital'] = 2

    # Columns with missing values:
    # + Income            3267 non-null float64
    # + Assets            3527 non-null float64
    # + Debt              3549 non-null float64
    # + ResidenceType     3558 non-null float64
    # + EmploymentType    3562 non-null float64

    raw.drop(['PhoneNumber', 'ApplicationId'], axis=1, inplace=True)

    # For check
    raw['Assets_origin'] = raw['Assets']

    # bad/(bad + good) in Assets NaN group is approximately equal to bad/(bad + good) in group with 0 Assets
    raw.loc[pd.isnull(raw['Assets']), 'Assets'] = 0

    raw.insert(len(raw.columns), 'AmountToPrice', (raw['Price'] - raw['Amount']) / raw['Price'])
    raw.loc[pd.isnull(raw['Debt']), 'Debt'] = 0
    raw.insert(len(raw.columns), 'debtToAssets', (raw['Debt'] - raw['Assets']) / (raw['Assets'] + 0.01), 1)
    raw.loc[raw['Debt'] > 0, 'Debt'] = 1

    # Only 1 record with Time 72
    raw.loc[raw['Time'] == 72, 'Time'] = 60
    # 13 records with Time 54 -> to Time 60
    raw.loc[raw['Time'] == 54, 'Time'] = 60
    # 25 records with Time 42 -> to Time 36
    raw.loc[raw['Time'] == 42, 'Time'] = 36
    # 30 records with Time 6 -> to Time 12
    raw.loc[raw['Time'] == 6, 'Time'] = 12

    # 2 is EmploymentType with the higher ratio of bad. Set for 2 records
    raw.loc[pd.isnull(raw['EmploymentType']), 'EmploymentType'] = 2
    raw['EmploymentType'] = raw['EmploymentType'].astype('int')

    raw = pd.concat([raw, pd.get_dummies(raw['EmploymentType'], prefix='EmploymentType')], axis=1)
    raw.drop(['EmploymentType'], axis=1, inplace=True)

    # Income not selected
    raw.insert(len(raw.columns), 'MissingIncome', np.where(np.isnan(raw['Income']), 1, 0))

    raw.loc[pd.isnull(raw['Income']), 'Income'] = 0
    raw.insert(len(raw.columns), 'RelIncomeToExpenses', raw['Income'] / (raw['Expenses'] + 0.1))

    raw['Assets'] = np.log(raw['Assets'] + 1)

    # Set avg ResidenceType
    mean_res_type = 3
    raw.loc[raw['ResidenceType'] == 4, 'ResidenceType'] = mean_res_type
    raw.loc[np.isnan(raw['ResidenceType']), 'ResidenceType'] = mean_res_type

    raw.drop(['Assets_origin'], axis=1, inplace=True)
    raw.drop(['Age', 'Expenses', 'Income', 'Assets', 'Debt', 'Amount'], axis=1, inplace=True)

    # show_heatmap(raw)

    return raw


df = format_data(df)
df_test = format_data(df_test)
df_test.drop(['Status'], axis=1, inplace=True)

# for cn in df.columns.values:
#     if cn != 'Status':
#         print(cn, df[cn].value_counts())


def _predict(_classifier, test_data):
    candidate_classifier = _classifier
    candidate_classifier.fit(train[0::, 1::], train[0::, 0])
    return candidate_classifier.predict(test_data)


def _predict_proba(_classifier, test_data):
    _classifier.fit(train[0::, 1::], train[0::, 0])
    return _classifier.predict_proba(test_data)


class EnsemblePredict:
    def __init__(self):
        self.p1 = LogisticRegression(random_state=42, C=10, penalty='l1')
        self.p2 = GradientBoostingClassifier(subsample=0.8, learning_rate=0.15,
                                             max_depth=3, min_samples_leaf=20, max_features=0.2)
        self.p3 = GaussianNB()
        self.p4 = RandomForestClassifier(random_state=42, max_features=0.3, n_estimators=70)

    def fit(self, _x_train, _y_train):
        self.p1.fit(_x_train, _y_train)
        self.p2.fit(_x_train, _y_train)
        self.p3.fit(_x_train, _y_train)
        self.p4.fit(_x_train, _y_train)

    def predict(self, data):
        res1 = _predict_proba(self.p1, data)
        res2 = _predict_proba(self.p2, data)
        res3 = _predict_proba(self.p3, data)
        res4 = _predict_proba(self.p4, data)

        return np.round((res1 + res2 + res3 + res4) / 4, 7)


classifiers = [
    EnsemblePredict(),
    KNeighborsClassifier(),
    SVC(probability=True, random_state=42),
    DecisionTreeClassifier(random_state=42),
    RandomForestClassifier(random_state=42, max_features=0.3, n_estimators=70),
    AdaBoostClassifier(random_state=42),
    # GradientBoostingClassifier(random_state=42,learning_rate=0.1,
    # max_features=0.1, max_depth=4, min_samples_leaf=20),
    # GradientBoostingClassifier(random_state=42, learning_rate=0.1,
    # max_features=0.1, max_depth=6, min_samples_leaf=100),
    GradientBoostingClassifier(subsample=0.8, learning_rate=0.15,
                               max_depth=3, min_samples_leaf=20, max_features=0.2),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression(random_state=42, C=10, penalty='l1')
]

num_of_test_splits = 8
stratified_split = StratifiedShuffleSplit(n_splits=num_of_test_splits, random_state=0)

train = df.values
col = df.copy()
col.drop(['Status'], axis=1, inplace=True)

X = train[0::, 1::]
y = train[0::, 0]


def fit_best():
    acc_dict = {}
    roc_dict = {}

    for train_index, test_index in stratified_split.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        for clf in classifiers:
            name = clf.__class__.__name__
            clf.fit(X_train, y_train)
            train_predictions = clf.predict(X_test)
            acc = accuracy_score(y_test, train_predictions)
            if name in acc_dict:
                acc_dict[name] += acc
            else:
                acc_dict[name] = acc

            ro = roc_auc_score(y_test, train_predictions)
            if name in roc_dict:
                roc_dict[name] += ro
            else:
                roc_dict[name] = ro

    for cl in acc_dict:
        acc_dict[cl] /= num_of_test_splits
        roc_dict[cl] /= num_of_test_splits

    print(acc_dict)
    print(roc_dict)


# Params tune section
def tune_params(clf, train_x, train_y, init_params, need_scale=False):

    if need_scale:
        train_x = MinMaxScaler().fit_transform(train_x)

    grid = GridSearchCV(estimator=clf, param_grid=init_params, scoring='roc_auc', cv=8)
    grid.fit(train_x, train_y)
    return grid.best_score_, grid.best_params_, grid.best_estimator_


# GradientBoostingClassifier init search params
gb_grid_params = {'learning_rate': [0.1, 0.05, 0.15],
                  'max_depth': [3, 4],
                  'min_samples_leaf': [20, 50],
                  'max_features': ['sqrt', 0.2, 0.15, 0.1],
                  'subsample': [0.75, 0.8, 0.85]
                  }

# RandomForestClassifier init search params
f_grid_params = {'n_estimators': [10, 20, 30, 50, 70],
                 'max_features': [None, 'sqrt', 0.3, 0.2, 0.1]}

# LogisticRegression init search params
lr_grid_params = {'penalty': ['l1', 'l2'],
                  'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                  'max_iter': [80, 100, 120, 150]}

# clf = GradientBoostingClassifier()
# print(tune_params(clf, X, y, gb_grid_params))
# 0.850533270587311, {'subsample': 0.8, 'learning_rate': 0.15, 'max_depth': 3,
# 'min_samples_leaf': 20, 'max_features': 0.2},

# clf = RandomForestClassifier()
# print(tune_params(clf, X, y, f_grid_params))
# 0.8426235005814257, {'max_features': 0.3, 'n_estimators': 70},

# clf = LogisticRegression()
# print(tune_params(clf, X, y, lr_grid_params))
# (0.8257336880563437, {'C': 10, 'penalty': 'l1', 'max_iter': 100}

# fit_best()

clf = EnsemblePredict()
clf.fit(X, y)
predictions = clf.predict(df_test)

df_test_original['Status'] = predictions
# print(df_test_original)

# Predict and save data
df_test_original.to_csv(dir_path + '/Test.csv', index=False)
