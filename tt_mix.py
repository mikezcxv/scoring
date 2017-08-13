import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns; sns.set(style="ticks", color_codes=True)
from sklearn.model_selection import KFold
from sklearn.svm import SVC
import xgboost as xgb
from xgboost.sklearn import XGBRegressor, XGBClassifier
from sklearn.metrics import accuracy_score, log_loss, r2_score
import matplotlib.pyplot as plt
from helpers.my_help import *
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import operator

dir_path = os.path.dirname(os.path.realpath(__file__))

train = pd.read_csv(dir_path + '/train.csv', header=0, dtype={'Age': np.float64})
test = pd.read_csv(dir_path + '/test.csv', header=0, dtype={'Age': np.float64})
full_data = [train, test]

# print(test.info())
# print(test.head())


def format_data(raw_data):
    prepared = pd.DataFrame(raw_data)
    prepared = prepared.drop(['PassengerId', 'Name', 'Ticket', 'Embarked', 'Cabin',
                              'Fare', 'Age'], axis=1)
    # Check influence of Name(mr ms), Cabin end Embarked
    prepared['Sex'] = np.where(prepared['Sex'] == 'male', 1, 0)

    # av = np.mean(data['Age'])
    # prepared['Age'] = prepared['Age'].fillna(av)
    # age_avg = prepared['Age'].mean()
    # age_std = prepared['Age'].std()
    # age_null_count = prepared['Age'].isnull().sum()
    # age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std,
    #                                          size=age_null_count)
    # prepared['Age'][np.isnan(prepared['Age'])] = age_null_random_list
    #
    # prepared.loc[prepared['Age'] <= 16, 'Age'] = 0
    # prepared.loc[(prepared['Age'] > 16) & (prepared['Age'] <= 32), 'Age'] = 1
    # prepared.loc[(prepared['Age'] > 32) & (prepared['Age'] <= 48), 'Age'] = 2
    # prepared.loc[(prepared['Age'] > 48) & (prepared['Age'] <= 64), 'Age'] = 3
    # prepared.loc[prepared['Age'] > 64, 'Age'] = 4
    # prepared['Age'] = prepared['Age'].astype(int)
    return prepared


data = train
data = format_data(data)

# xgb_params = {
#     # 'n_trees': 500,
#     'eta': 0.01,
#     'max_depth': 4,
#     'subsample': 0.95,
#     'colsample_bytree': 0.7,
#     'objective': 'binary:logistic',
#     'eval_metric': 'logloss',
#     'silent': 1,
#     # 'booster': 'gbtree',
# }

xgb_params = {
    'min_child_weight': 10.0,
    'objective': 'binary:logistic',
    'max_depth': 5,
    'eval_metric': 'auc',
    # 'max_delta_step': 1.8,
    'colsample_bytree': 0.4,
    'subsample': 0.7,
    'eta': 0.01,
    'gamma': 0.65
}

# xgb_params.update({
#     'max_depth': 5,
#     'eta': 0.023817282528656166,
#     'colsample_bytree': 0.5245332440974132,
#     'subsample': 0.764756590844847
# })

train.drop(['Name'], axis=1, inplace=True)
test.drop(['Name'], axis=1, inplace=True)


def cast_object(df):
    # Deal with categorical values
    df_numeric = df.select_dtypes(exclude=['object'])
    df_obj = df.select_dtypes(include=['object']).copy()

    for c in df_obj:
        df_obj[c] = pd.factorize(df_obj[c])[0]

    return pd.concat([df_numeric, df_obj], axis=1)


train = cast_object(train)
test = cast_object(test)

y_train = train[['Survived']]
X_train = train.drop(['Ticket', 'Fare', 'Survived', 'PassengerId'], axis=1)
id_test = test['PassengerId']
X_test = test.drop(['Ticket', 'Fare', 'PassengerId'], axis=1)

df_columns = train.columns
dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb.DMatrix(X_test)


def get_max_boost(print_cv=1):
    cv_result = xgb.cv(xgb_params, dtrain, num_boost_round=2000,
                       early_stopping_rounds=50, verbose_eval=50, show_stdv=False,
                       )

    # cv_result[['train-logloss-mean', 'test-logloss-mean']].plot()
    # plt.show()
    num_boost_rounds = len(cv_result)
    if print_cv:
        print(cv_result)

    print(num_boost_rounds)

    return num_boost_rounds


# nbr = 424


def cv2(xgb_params, train, y_train, num_boost_rounds=474):
    X = train.values
    y = y_train.values
    kf = KFold(n_splits=5)
    kf.get_n_splits(X)

    i = 0
    r2 = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        print("Set:", i, ':')
        i += 1

        dtrain = xgb.DMatrix(X_train, y_train)
        dtest = xgb.DMatrix(X_test)
        clf = xgb.train(dict(xgb_params, silent=1), dtrain,
                        num_boost_round=num_boost_rounds, verbose_eval=0)

        y_predicted = np.round(clf.predict(dtest))
        s = accuracy_score(y_test, y_predicted)
        print(s)
        r2.append(s)

    print('AVG:', sum(r2) / 5)

    return sum(r2) / 5


def go(num_boost_round, submission_index, show_graph=1):
    params = dict(xgb_params, silent=1)
    model = xgb.train(params, dtrain, num_boost_round=num_boost_round)

    print(model.feature_names)
    print(len(model.feature_names))
    print(model.get_fscore())
    print(len(model.get_fscore()))

    scores = model.get_fscore()
    list_useless = []
    almost_useless = []
    for fn in model.feature_names:
        if fn not in scores:
            list_useless.append(fn)
        elif scores[fn] < 50:
            almost_useless.append(fn)

    print(list_useless)
    print(almost_useless)

    if show_graph:
        fig, ax = plt.subplots(1, 1, figsize=(8, 16))
        xgb.plot_importance(model, height=0.5, ax=ax)
        plt.show()

    y_pred = model.predict(dtest)
    y_pred = np.round(model.predict(dtest))
    y_pred = y_pred.astype(int)
    result = pd.DataFrame({'PassengerId': id_test, 'Survived': y_pred})
    result.to_csv('sub' + submission_index + '.csv', index=False)


def objective(space):
    num_boost_round = 1000
    xgb_params = {
        # 'n_trees': int(space['n_trees']),
        'eta': float(space['eta']),
        'max_depth': int(space['max_depth']),
        'subsample': float(space['subsample']),
        'colsample_bytree': float(space['colsample_bytree']),
        'objective': 'reg:linear',
        'eval_metric': 'rmse'
    }

    cv_av_result = cv2(xgb_params, X_train, y_train)

    print(xgb_params)
    print("Score:", cv_av_result)
    return {'loss': 1 - cv_av_result, 'status': STATUS_OK}


space = {
    'max_depth': hp.quniform("x_max_depth", 3, 5, 1),
    'subsample': hp.uniform('x_subsample', 0.65, 0.95),
    'colsample_bytree': hp.uniform('x_colsample_bytree', 0.45, 0.75),
    'eta': hp.uniform('x_eta', 0.001, 0.3)
}


def train_params():
    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials)
    print(best)
    asd()


# cv2(xgb_params, X_train, y_train, 474)
# train_params()
# asd()


# show_values_stat(train, 'Embarked')
# print(stats.ks_2samp(train['Embarked'], train['Survived']))
# show_pair_plot(train, ['Embarked', 'Survived'])
# asd()

# print(train[['Ticket']])
# asd()

# show_pair_plot(train, ['Ticket', 'Survived'])
# show_pair_correlations(train, df_columns)
# asd()
# get_max_boost(1)
# asd()
# cv2(xgb_params, train, y_train, 98)
# asd()

go(94, '31')
asd()

# cv2()

def _predict(_classifier, test_data):
    candidate_classifier = _classifier
    candidate_classifier.fit(train[0::, 1::], train[0::, 0])
    return candidate_classifier.predict(test_data)

train = data.values

test = format_data(test)
result_svc = _predict(SVC(), test)
full_data[1]['Survived'] = result_svc

submit = full_data[1][['PassengerId', 'Survived']]

submission = pd.DataFrame({'PassengerId': full_data[1][['PassengerId']],
                           'Survived': result_svc})

print(submit.to_csv("submission.csv", index=False))

# print(result_svc[0,])
# print(test.head())

