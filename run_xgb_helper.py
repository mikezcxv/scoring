import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import numpy as np
import pandas as pd
import math
import re
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy import stats as stats1
import statsmodels.api as sm
import statsmodels.formula.api as smf
from ggplot import *
from sklearn.model_selection import StratifiedShuffleSplit, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, auc, roc_auc_score, roc_curve, classification_report
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split
from helper.my_helper import *
import random
import pickle
from tqdm import tqdm


def split_into_folds(df, target, random_state=42):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    res = []
    for big_ind, small_ind in skf.split(np.zeros(len(df)), df[target]):
        res.append((df.iloc[big_ind], df.iloc[small_ind]))


class RunXGB:
    def __init__(self, xgb_params, df, df_train, target_train, df_test, target_test, id_test, target, pk,
                 num_of_test_splits=5):
        self.xgb_params = xgb_params
        self.target = target
        self.pk = pk
        self.df = df
        self.num_of_test_splits = num_of_test_splits
        self.df_train = df_train
        self.df_test = df_test
        self.target_train = target_train
        self.target_test = target_test
        self.dtrain = xgb.DMatrix(df_train, target_train)
        self.dtest = xgb.DMatrix(df_test)
        self.id_test = id_test

    def get_max_boost(self, debug=False, num_folds=6, max_boost_rounds=1500, verbose_eval=50, count_extra_run=1):
        # Data structure in which to save out-of-folds preds
        early_stopping_rounds = 30

        results = {'test-auc-mean': [], 'test-auc-std': [], 'train-auc-mean': [], 'train-auc-std': [],
                   'num_rounds': []}
        iter_cv_result = []
        for i in tqdm(range(count_extra_run)):
            verb = verbose_eval if i < 2 else None
            iter_cv_result.append(xgb.cv(dict(self.xgb_params, silent=1, seed=i + 1), self.dtrain,
                                         num_boost_round=max_boost_rounds,
                                         early_stopping_rounds=early_stopping_rounds,
                                         verbose_eval=verb, show_stdv=False,
                                         metrics={'auc'}, stratified=False, nfold=num_folds))

            results['num_rounds'].append(len(iter_cv_result[i]))
            t = iter_cv_result[i].ix[results['num_rounds'][i] - 1, :]

            for c in ['test-auc-mean', 'test-auc-std', 'train-auc-mean', 'train-auc-std']:
                results[c].append(t[c])

        num_boost_rounds = np.mean(results['num_rounds'])

        # Show results
        if debug:
            for c in ['test-auc-mean', 'test-auc-std', 'train-auc-mean', 'train-auc-std', 'num_rounds']:
                factor = 100 if c != 'num_rounds' else 1
                print('Avg %s: %.2f +- %.2f; [%.2f .. %.2f]; range: %.2f; median: %.2f' %
                      (c, np.mean(results[c]) * factor, np.std(results[c]) * factor, np.min(results[c]) * factor,
                       np.max(results[c]) * factor, (np.max(results[c]) - np.min(results[c])) * factor,
                       np.median(results[c]) * factor))

            # print('Results of the first round', iter_cv_result[0])
            print('Best iteration:', num_boost_rounds)

        return int(num_boost_rounds)

    def get_max_boost_by_last_periods(self, debug=False, count_last_periods=3):
        #         kf = KFold(n_splits=self.num_of_test_splits)
        #         kf.get_n_splits(self.df_train)

        kf = KFold(n_splits=self.num_of_test_splits)
        l = kf.get_n_splits(self.df_train)
        g = kf.split(self.df_train)
        f = []
        for i in range(0, self.num_of_test_splits - count_last_periods):
            next(g)

        for i in range(0, count_last_periods):
            f.append(next(g))

        # kf.split(self.df_train)
        # for train_index, test_index in kf.split(self.df_train):

        cv_result = xgb.cv(dict(self.xgb_params, silent=1), self.dtrain, num_boost_round=1500,
                           early_stopping_rounds=50, verbose_eval=50, show_stdv=False,
                           metrics={'auc'}, stratified=False, nfold=self.num_of_test_splits,
                           folds=f)

        # cv_result[['train-logloss-mean', 'test-logloss-mean']].plot()
        # plt.show()
        num_boost_rounds = len(cv_result)
        if debug:
            print(cv_result)
            print(num_boost_rounds)

        return num_boost_rounds

    def load_model(self, model_full_path='data/fraud_model.dat'):
        return pickle.load(open(model_full_path, "rb"))

    def go_and_save_model(self, num_boost_round, p_o_s, target, show_graph=True, threshold_useless=3,
                          files_prefix='', debug=True, show_auc=True,
                          model_full_path='data/fraud_model.dat'):
        params = dict(self.xgb_params, silent=1)

        df = self.df.copy()
        df = df.iloc[:math.ceil(len(df) * p_o_s), :]

        df_train = df.ix[:, df.columns != target]
        target_train = df.ix[:, target]
        id_train = df_train.ix[:, self.pk]
        df_train.drop([self.pk], axis=1, inplace=True)
        df_columns = df.columns
        dtrain = xgb.DMatrix(df_train, target_train)

        print(df_train.shape)

        model = xgb.train(params, dtrain, num_boost_round=num_boost_round)

        # save model to file
        pickle.dump(model, open(model_full_path, "wb"))

        files_prefix = '_' + files_prefix

        if debug:
            print("Using %d features" % len(model.feature_names))
            print(model.get_fscore(), len(model.get_fscore()))

        scores = model.get_fscore()
        list_useless = []
        almost_useless = []
        for fn in model.feature_names:
            if fn not in scores:
                list_useless.append(fn)
            elif scores[fn] < threshold_useless:
                almost_useless.append(fn)

        if debug:
            print("List useless:", list_useless)
            print("List almost useless:", almost_useless)

        if show_graph:
            fig, ax = plt.subplots(1, 1, figsize=(8, 16))
            xgb.plot_importance(model, height=0.5, ax=ax)
            plt.show()
            plt.savefig('imgs/all_data' + files_prefix + 'features.png')

    def go(self, num_boost_round, show_graph=True, threshold_useless=3, files_prefix='', debug=True, show_auc=True):
        params = dict(self.xgb_params, silent=1)
        model = xgb.train(params, self.dtrain, num_boost_round=num_boost_round)
        files_prefix = '_' + files_prefix

        # print(model.feature_names)
        if debug:
            print("Using %d features" % len(model.feature_names))
            print(model.get_fscore())
            print(len(model.get_fscore()))

        scores = model.get_fscore()
        list_useless = []
        almost_useless = []
        for fn in model.feature_names:
            if fn not in scores:
                list_useless.append(fn)
            elif scores[fn] < threshold_useless:
                almost_useless.append(fn)

        if debug:
            print("List useless:")
            print(list_useless)
            print("List almost useless:")
            print(almost_useless)

        if show_graph:
            fig, ax = plt.subplots(1, 1, figsize=(8, 16))
            xgb.plot_importance(model, height=0.5, ax=ax)
            plt.show()
            plt.savefig('imgs/' + files_prefix + 'features.png')

        y_pred = model.predict(self.dtest)
        _acc = accuracy_score(self.target_test, np.round(y_pred))
        _roc = roc_auc_score(self.target_test, y_pred)

        # false_positive_rate, true_positive_rate, thresholds = roc_curve(target_test, y_pred)
        # may differ
        # print auc(false_positive_rate, true_positive_rate)

        if show_auc:
            fpr, tpr, _ = roc_curve(self.target_test, y_pred)

            # Plot of a ROC curve for a specific class
            plt.figure()
            plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % _roc)
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc="lower right")
            plt.savefig('imgs/' + files_prefix + 'aucc.png')
            plt.clf()

        print("Accuracy on hold-out: %.2f" % _acc)
        print("Roc on hold-out: %.3f" % _roc)

        result = pd.DataFrame({'id': self.id_test, self.target: y_pred, 'realVal': self.target_test})

        if show_graph:
            plt.hist(result.loc[result.realVal == 1, self.target], 100, alpha=0.8, label='Bad', color="red")
            plt.hist(result.loc[result.realVal == 0, self.target], 100, alpha=0.5, label='Good')
            plt.legend(loc='upper right')
            # pyplot.show()
            plt.savefig('imgs/' + files_prefix + 'distributions.png')

        result.to_csv('data/' + files_prefix + 'result.csv', index=False)
        return _roc

    def go_multi(self, num_boost_round, columns_sets, show_graph=True, threshold_useless=3,
                 files_prefix='', debug=True, show_auc=True, count_rounds=10, min_auc=0.660):

        y_preds = []
        files_prefix = '_' + files_prefix

        #         columns_sets = [only_columns_v1, only_columns_last, core_columns, only_columns_adjusted]

        med_result = pd.DataFrame({'id': self.id_test, 'realVal': self.target_test})
        #         for j in range(0, count_rounds):

        for j in range(0, len(columns_sets)):

            # For random sampling
            # sub_cols = random.sample(set(self.df_train.columns.values), 70)
            #             sub_cols = columns_sets[j]

            df_train_with_reap = self.df_train.sample(frac=1, replace=True, random_state=j, axis=0)
            target_train_with_reap = self.target_train.ix[df_train_with_reap.index]

            sub_cols = [x for x in last_run_columns if (x not in [target, pk])]
            dtrain = xgb.DMatrix(df_train_with_reap[sub_cols], target_train_with_reap)
            dtest = xgb.DMatrix(self.df_test[sub_cols])

            params = dict(self.xgb_params, silent=1,
                          scale_pos_weight=len(target_train_with_reap) / target_train_with_reap.sum())

            cv_result = xgb.cv(params, dtrain, num_boost_round=120, early_stopping_rounds=50,
                               verbose_eval=50, show_stdv=False, metrics={'auc'}, stratified=False, nfold=5)

            test_cv = cv_result['test-auc-mean'].iloc[-1]
            print("Test cv: %.3f" % test_cv)

            if test_cv > min_auc:
                model = xgb.train(params, dtrain, num_boost_round=round(num_boost_round * 1.2))

                # print(model.feature_names)
                if debug:
                    print("Using %d features" % len(model.feature_names))
                    print(model.get_fscore())
                    print(len(model.get_fscore()))

                scores = model.get_fscore()
                list_useless = []
                almost_useless = []
                for fn in model.feature_names:
                    if fn not in scores:
                        list_useless.append(fn)
                    elif scores[fn] < threshold_useless:
                        almost_useless.append(fn)

                if debug:
                    print("List useless:")
                    print(list_useless)
                    print("List almost useless:")
                    print(almost_useless)

                if show_graph:
                    fig, ax = plt.subplots(1, 1, figsize=(8, 16))
                    xgb.plot_importance(model, height=0.5, ax=ax)
                    plt.show()
                    plt.savefig('imgs/' + files_prefix + 'features.png')

                y_pred = model.predict(dtest)
                y_preds.append(y_pred)
                med_result['target_' + str(j)] = y_pred

        for i in range(0, len(y_preds) - 1):
            print(i)
            y_pred += y_preds[i]

        y_pred /= len(y_preds)

        _roc = roc_auc_score(self.target_test, y_pred)

        corr_matrix = med_result.corr()
        print(corr_matrix)
        plt.figure()
        ax = sns.heatmap(corr_matrix, cmap='bwr')
        plt.savefig("imgs/" + files_prefix + "_corr_matrix_results.png")
        plt.clf()

        # false_positive_rate, true_positive_rate, thresholds = roc_curve(target_test, y_pred)
        # may differ
        # print auc(false_positive_rate, true_positive_rate)

        if show_auc:
            fpr, tpr, _ = roc_curve(self.target_test, y_pred)

            # Plot of a ROC curve for a specific class
            plt.figure()
            plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % _roc)
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc="lower right")
            plt.savefig('imgs/' + files_prefix + 'aucc.png')
            plt.clf()

        # print("Accuracy on hold-out: %.2f" % _acc)
        print("Roc on hold-out: %.3f" % _roc)

        result = pd.DataFrame({'id': self.id_test, self.target: y_pred, 'realVal': self.target_test})

        #         if show_graph:
        plt.hist(result.loc[result.realVal == 1, self.target], 100, alpha=0.8, label='Bad', color="red")
        plt.hist(result.loc[result.realVal == 0, self.target], 100, alpha=0.5, label='Good')
        plt.legend(loc='upper right')
        # pyplot.show()
        plt.savefig('imgs/' + files_prefix + 'distributions.png')

        result.to_csv('data/' + files_prefix + 'result.csv', index=False)
        return _roc

    def cv_xgb(self, xgb_params, train, y_train, num_boost_rounds=474, split_useless_bottom=2):
        # stratified_split = StratifiedShuffleSplit(n_splits=self.num_of_test_splits, random_state=0)
        acc_overall = 0
        roc_overall = 0
        X = train.values
        y = y_train.values
        kf = KFold(n_splits=self.num_of_test_splits)
        kf.get_n_splits(X)

        i = 0
        list_useless = {}
        list_alm = {}
        l_f = {}
        #    for train_index, test_index in stratified_split.split(X, y):

        #         Cont.df_res = df_train[[pk]].copy()
        #         Cont.df_res['scores'] = np.NAN
        #         Cont.df_res['real_y'] = np.NAN

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            print("Set:", i, ':')
            i += 1

            dtrain = xgb.DMatrix(X_train, y_train, feature_names=train.columns.values)
            dtest = xgb.DMatrix(X_test, feature_names=train.columns.values)
            model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds, verbose_eval=0)
            train_predictions = model.predict(dtest)

            #             print(test_index)

            #             Cont.df_res.ix[test_index, 'scores'] = train_predictions
            #             Cont.df_res.ix[test_index, 'real_y'] = y_test

            #             print(Cont.df_res)

            _acc = accuracy_score(y_test, np.round(train_predictions))
            _roc = roc_auc_score(y_test, train_predictions)

            # false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, clf.predict(dtest))
            # may differ
            # print(auc(false_positive_rate, true_positive_rate))

            print(_acc)
            print(_roc)
            list_useless[i], list_alm[i] = report_useless_features(model, split_useless_bottom)
            l_f[i] = np.unique(list_useless[i] + list_alm[i]).tolist()
            # list(set(list_useless[i] + list_alm[i]))

            acc_overall += _acc
            roc_overall += _roc

        acc_overall /= self.num_of_test_splits
        roc_overall /= self.num_of_test_splits

        #         print(predictions, original)

        print("Avg accuracy %.2f " % acc_overall)
        print("Avg roc %.2f " % roc_overall)
        print("Useless in all periods:")
        # list_useless[1], list_useless[2], list_useless[3],
        useless = find_lists_intersection(list_useless[self.num_of_test_splits - 2],
                                          list_useless[self.num_of_test_splits - 1],
                                          list_useless[self.num_of_test_splits])
        print(useless)
        print("Almost useless(< 2 splits) in all periods:")
        # l_f[1], l_f[2], l_f[3],
        almost_u = find_lists_intersection(l_f[self.num_of_test_splits - 2],
                                           l_f[self.num_of_test_splits - 1], l_f[self.num_of_test_splits]
                                           )
        print(almost_u)
        #         print(Cont.df_res)

        return [useless, almost_u]

    # With wait_useless_eliminate=True run untill all useless are empty
    def cv_chain(self, max_rounds, max_cv_rounds=2, min_split_prun=0, debug=True):
        useless, tmp = self.cv_xgb(self.xgb_params, self.df_train, self.target_train, max_rounds, min_split_prun + 1)

        i = 0
        usefull_features = []
        while len(useless) and i < max_cv_rounds - 1:
            i += 1
            try:
                df_train_copy
            except NameError:
                df_train_copy = self.df_train.copy()
                if min_split_prun > 0:
                    useless = tmp

            df_train_copy.drop(useless, axis=1, inplace=True)
            usefull_features = df_train_copy.columns.values
            if debug:
                print("Full CV round: %d; Features left: %d" % (i, len(df_train_copy.columns.values)))

            if min_split_prun > 0:
                tmp, useless = self.cv_xgb(self.xgb_params, df_train_copy, self.target_train, max_rounds,
                                           min_split_prun + 1)
            else:
                useless, almos_u = self.cv_xgb(self.xgb_params, df_train_copy, self.target_train, max_rounds)

        return useless, usefull_features
