import sys
import math
import pandas as pd
import numpy as np
# import missingno as msno
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
sns.set(style="ticks", color_codes=True)


def get_ols(data, features):
    # cw_lm1 = ols('price_doc ~ ' + '*'.join(features), data=data, missing='drop').fit()
    original_len = len(data)
    data_copy = data
    # for f in features:
    data_copy = data_copy[
        ~np.isnan(data_copy[features[0]])
        & ~np.isnan(data_copy[features[1]])
    ]

    cleaned_len = len(data_copy)

    # '+'.join(features) + ' + ' +
    cw_lm1 = ols('price_doc ~ ' + '*'.join(features), data=data_copy, missing='drop').fit()
    r = sm.stats.anova_lm(cw_lm1, typ=2)

    r1 = r.iloc[0]
    r2 = r.iloc[1]
    r1_2 = r.iloc[2]
    r3 = r.iloc[3]
    r1_3 = r.iloc[4]
    r2_3 = r.iloc[5]
    r1_2_3 = r.iloc[6]

    if r1['F'] > r3['F'] or r2['F'] > r3['F'] \
            or (r1_2['F'] > r1['F'] and r1_2['F'] > r2['F']) or r1_3['F'] > r3['F'] \
            or r1_2_3['F'] > r1_3['F'] > max(r1['F'], r3['F'])\
            or r1_2_3['F'] > r2_3['F'] > max(r2['F'], r3['F']):
        if original_len != cleaned_len:
            print('Before: %d; After: %d; Lost: %2.2f%%' %
                  (original_len, cleaned_len,
                   round((original_len - cleaned_len) * 100 / original_len, 2)))
        return r
    else:
        return None

    # f1 = r.iloc[0]['F']
    # p1 = r.iloc[0]['PR(>F)']
    # f2 = r.iloc[1]['F']
    # p2 = r.iloc[1]['PR(>F)']
    # f3 = r.iloc[2]['F']
    # p3 = r.iloc[2]['PR(>F)']

    # if (p3 < .01) and (f3 > 50) and (f3 > f2) and (f3 > f1):

    # else:
    #     return None


def show_correlation(data, feature1, feature2):
    print(stats.pearsonr(data[[feature1]], data[[feature2]]))
    # print(stats.kendalltau(data[[feature1]], data[[feature2]]))
    # print(stats.spearmanr(data[[feature1]], data[[feature2]]))


def show_c_3_2_correlation(data, feature1, feature2, y = 'price_doc'):
    print(feature1, ' : ', feature2, stats.pearsonr(data[[feature1]], data[[feature2]]))
    print(feature1, ' : ', y, stats.pearsonr(data[[feature1]], data[[y]]))
    print(feature2, ' : ', y, stats.pearsonr(data[[feature2]], data[[y]]))


def show_c_3_2_correlation_spearman(data, feature1, feature2, y = 'price_doc'):
    print(feature1, ' : ', feature2, stats.spearmanr(data[[feature1]], data[[feature2]]))
    print(feature1, ' : ', y, stats.spearmanr(data[[feature1]], data[[y]]))
    print(feature2, ' : ', y, stats.spearmanr(data[[feature2]], data[[y]]))


def show_ks_stat(data1, data2, feature):
    r = stats.ks_2samp(data1[feature], data2[feature])
    if r.pvalue < 0.01:
        print('Looks like we have different distributions')
    elif r.pvalue < 0.1:
        print('Looks like we have slightly different distributions')
    else:
        print('Looks like we have identical distributions')

    print(r)


def compare_binary(data, feature1, low_val=0, hi_val=1, target='y'):
    l = data.loc[data[feature1] == low_val]
    h = data.loc[data[feature1] == hi_val]

    print('Count low:', len(l), 'Count hi:', len(h))

    print('Mean low:', np.mean(l[target]))
    print('Mean hi:', np.mean(h[target]))

    r = stats.ks_2samp(l[target], h[target])
    if r.pvalue < 0.01:
        print('Looks like we have different distributions')
    elif r.pvalue < 0.1:
        print('Looks like we have slightly different distributions')
    else:
        print('Looks like we have identical distributions')

    print(r)


def show_correlations(data, mutable_feature, feature='price_doc', hide_funcs=0):
    if not hide_funcs:
        var = data[[mutable_feature, feature]]
        var[mutable_feature + '_log_e'] = np.log(1 + data[mutable_feature])
        var[mutable_feature + '_log_10'] = np.log10(1 + data[mutable_feature])
        var[mutable_feature + '_x_3'] = data[mutable_feature] ** -3
        var[mutable_feature + '_x_2'] = data[mutable_feature] ** -2
        var[mutable_feature + '_x_1'] = data[mutable_feature] ** -1
        var[mutable_feature + '_x2'] = data[mutable_feature] ** 2
        var[mutable_feature + '_x3'] = data[mutable_feature] ** 3
        var[mutable_feature + '_x4'] = data[mutable_feature] ** 4

    print('Feature: %s' % mutable_feature)
    print(stats.kendalltau(data[[mutable_feature]], data[[feature]]))
    print(stats.spearmanr(data[[mutable_feature]], data[[feature]]))
    print('Pearson: ')
    show_correlation(data, mutable_feature, feature)

    if not hide_funcs:
        list_funcs = ['log_e', 'log_10', 'x_3', 'x_2', 'x_1', 'x2', 'x3', 'x4']
        for f in list_funcs:
            print(f, ': ', end='')
            show_correlation(var, mutable_feature + '_' + f, feature)


def show_type_grouped(df):
    print(df.columns.to_series().groupby(df.dtypes).groups)


def show_binary_comparison(df, feat_name, by_fea = 'y', left=0, right=1):
    print(df.groupby(feat_name, as_index=False)[by_fea]
          .agg(['count', 'mean', 'min', 'max']))

    set1 = df[df[feat_name] == right][by_fea]
    set2 = df[df[feat_name] == left][by_fea]
    print(stats.ks_2samp(set1, set2))

    original1, original2 = len(set1), len(set2)

    set1 = set1[~np.isnan(set1)]
    set2 = set2[~np.isnan(set2)]

    after_rm1, after_rm2 = len(set1), len(set2)

    if after_rm1 < original1:
        print('%d values removed from set 1!' % (original1 - after_rm1))

    if after_rm2 < original2:
        print('%d values removed from set 2!' % (original2 - after_rm2))

    print(stats.ttest_ind(set1, set2))


def get_normalized_std(df, feature):
    l = df[~np.isnan(df[feature])]
    return np.std(l[feature] / np.linalg.norm(l[feature]))


def show_values_stat(df, feat_name):
    print(df.groupby(feat_name, as_index=False)[feat_name]
          .agg(['count']))


def show_pair_plot(data, features, hue=None):
    cp = data.copy()
    print('Original len:', len(cp))
    for f in features:
        cp = cp[~np.isnan(cp[f])]

    print('Len after missing clean up:', len(cp))

    sns.pairplot(cp[features], kind="reg", hue=hue)
    plt.show()


def show_pair_correlations(df, data, sensitivity=.7, corr_var=None):
    i = 0
    li = []
    for a in data:
        i += 1
        j = 0
        for b in data:
            j += 1
            # print(i, j, end='; ')
            if a != b and j < i:

                # print(a, b)
                s = stats.spearmanr(df[[a]], df[[b]])
                if abs(s.correlation) > sensitivity:
                    print(a, b, s)
                    if corr_var:
                        sp = stats.spearmanr(df[[a]], df[[corr_var]])
                        print("\t" + a, sp)
                        sp = stats.spearmanr(df[[b]], df[[corr_var]])
                        print("\t" + b, sp)
                    if a not in li:
                        li.append(a)
                    # if b not in li:
                    #     li.append(b)
    return li


def show_bar_plot(data, feature_x, feature_y):
    sns.barplot(x=feature_x, y=feature_y, data=data, )
    plt.show()


def show_scatter_plot(data, feature_x, feature_y):
    plt.scatter(x=feature_x, y=feature_y, data=data)
    plt.show()


def show_scatter_multi_plot(data, feature_x, feature_y, feature_x2):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.scatter(x=feature_x, y=feature_y, data=data, marker="s", label=feature_x)
    ax1.scatter(x=feature_x2, y=feature_y, data=data, marker="o", label=feature_x2)
    plt.legend(loc='upper left')
    plt.show()


def show_hist(l, bins='auto'):
    l = l[~np.isnan(l)]
    plt.hist(l, bins=bins)
    plt.show()


def show_hist_multi(df1, df2, feature, bins='auto'):
    df1 = df1[~np.isnan(df1[feature])]
    df2 = df2[~np.isnan(df2[feature])]
    plt.hist(df1[feature], bins)
    plt.hist(df2[feature], bins)
    plt.show()


def show_feature_over_time(df, feature):
    df['date_column'] = pd.to_datetime(df['timestamp'])
    df['mnth_yr'] = df['date_column'].apply(lambda x: x.strftime('%B-%Y'))

    df = df[[feature,"mnth_yr"]]
    df_vis = df.groupby('mnth_yr')[feature].mean()
    df_vis = df_vis.reset_index()
    df_vis['mnth_yr'] = pd.to_datetime(df_vis['mnth_yr'])
    df_vis.sort_values(by='mnth_yr')
    df_vis.plot(x='mnth_yr', y=feature)
    plt.show()


def describe(raw_data, varname="price_doc", minval=-1e20,
             maxval=1e20, addtolog=1, nlo=8, nhi=8, dohist=True, showmiss=True,
             show_left=None, show_right=None):
    var = raw_data[varname]

    info_fields = ['sub_area', 'price_doc', 'kremlin_km', 'mkad_km',
                   'full_sq', 'life_sq', 'num_room', 'kitch_sq', 'build_year',
                   'product_type']

    print("DESCRIPTION OF [", varname, "]\n")
    if showmiss:
        print("Fraction missing = ", var.isnull().mean(), "\n")

    var = var[(var <= maxval) & (var >= minval)]
    if nlo > 0:
        print("Lowest values:\n", var.sort_values().head(nlo).values, "\n")
        if show_left:
            print(raw_data.sort_values(varname).head(nlo)[info_fields])

    if nhi > 0:
        print("Highest values:\n", var.sort_values().tail(nhi).values, "\n")
        if show_right:
            print(raw_data.sort_values(varname).tail(nlo)[info_fields])

    if dohist:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 3))

        print("Histograms of raw values and logarithm")
        var.plot(ax=axes[0], kind='hist', bins=100)
        np.log(var + addtolog).plot(ax=axes[1], kind='hist', bins=100, color='green', secondary_y=True)
        plt.show()


def check(data, column_name):
    print(data.groupby([column_name, 'price_doc'])['price_doc'].mean())
    print('Grouped ')
    print(data.groupby([column_name])[column_name].count())
    print('Mean price ')
    print(data.groupby([column_name])['price_doc'].mean())
    print('Max price ')
    print(data.groupby([column_name])['price_doc'].max())
    print('Min price ')
    print(data.groupby([column_name])['price_doc'].min())


def check_outlines(data, field_name, show_left=None, show_right=None):
    rlimit = np.percentile(data[field_name].values, 99.5)
    llimit = np.percentile(data[field_name].values, 0.5)
    print('[', rlimit, llimit, ']',
          'Total: ', len(data),
          'Left: ',  len(data[data[field_name] < llimit]),
          'Right: ', len(data[data[field_name] > rlimit]))

    if show_left:
        print(data[data[field_name] < llimit][['sub_area', 'price_doc', 'kremlin_km', 'mkad_km']])

    if show_right:
        print(data[data[field_name] > rlimit][['sub_area', 'price_doc', 'kremlin_km', 'mkad_km']])


def check_nan(data, field_name):
    print(len(data))
    print(len(data[np.isnan(data[field_name])]))


def check_mean(data, field_name):
    print(data[['price_doc', field_name]].groupby([field_name], as_index=False)
          .agg(['mean', 'count']))


def describe_my(data, field_name):
    print(data[field_name].value_counts())
    check_mean(data, field_name)
    check_nan(data, field_name)


def set_av(data, filed_name):
    data[filed_name] = data[filed_name].fillna(data[filed_name].mean())


def set_default(data, filed_name, default_val):
    data[filed_name] = data[filed_name].fillna(default_val)


def percentile_validate(data, feature_name, debug=None, custom_default=None,
                        skip_left_outlines=None, add_extra_field=None,
                        r_percent=99.5, l_percent=0.5):
    rlimit = np.percentile(data[feature_name].values, r_percent)
    llimit = np.percentile(data[feature_name].values, l_percent)

    if add_extra_field:
        data.loc[data[feature_name] > rlimit, 'more_' + feature_name] \
            = data[feature_name]

    if debug:
        print('[', llimit, rlimit, ']')
        # print('Na: ', len(data[np.isnan(data[feature_name])]))

    default_validate(data, feature_name, llimit, rlimit, custom_default,
                     skip_left_outlines)


def default_validate(data, feature_name, _min, _max, custom_default=None,
                     skip_left_outlines=None):
    if skip_left_outlines:
        data.loc[data[feature_name] < _min, feature_name] = np.NaN
    else:
        data.loc[data[feature_name] < _min, feature_name] = _min

    data.loc[data[feature_name] > _max, feature_name] = _max
    if custom_default:
        set_default(data, feature_name, custom_default)
    else:
        set_av(data, feature_name)
    # data[feature_name] = data[feature_name].astype(int)


def show_missing_values(data):
    # Number of missing at each column
    missing_df = data.isnull().sum(axis=0).reset_index()
    missing_df.columns = ['column_name', 'missing_count']
    missing_df = missing_df.ix[missing_df['missing_count'] > 0]
    ind = np.arange(missing_df.shape[0])
    fig, ax = plt.subplots(figsize=(12, 18))
    ax.barh(ind, missing_df.missing_count.values, color='y')
    ax.set_yticks(ind)
    ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
    ax.set_xlabel("Count of missing values")
    ax.set_title("Number of missing values in each column")
    plt.show()


def show_missing_values_advanced(data):
    missingValueColumns = data.columns[data.isnull().any()].tolist()
    # msno.bar(train[missingValueColumns],
    #          figsize=(20, 8), color=(0.5, 0.5, 1), fontsize=12, labels=True, )
    msno.matrix(data[missingValueColumns], width_ratios=(10, 1),
                figsize=(20, 8), color=(0.5, 0.5, 1), fontsize=12, sparkline=True, labels=True)


def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i, pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0 / len(y))) ** 0.5


def rmsle_vectorized(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))


def asd():
    sys.exit()


def pr(*s):
    print(s)


def show_data_types(df):
    print(len(df.columns))
    print(df.dtypes.to_dict())
    print(df.dtypes.value_counts())


def cast_object_column(df, name):
    df[name] = df[name].astype('category')
    df[name] = df[name].cat.codes


def cast_object_column_gen2(df, name):
    df_train_cut = df_train.drop(['y'], axis=1)
    df_all_for_categorical = pd.concat([df_train_cut, df_test])

    for o in ['X0', 'X1', 'X2', 'X3', 'X5', 'X6', 'X8']:
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(df_all_for_categorical[o])
        df_train['mod'] = label_encoder.transform(df_train[o])
        df_test['mod'] = label_encoder.transform(df_test[o])

        print(df_train[['X0', 'mod']].head())
        print(df_test[['X0', 'mod']].head())

        # encode string class values as integers
    # label_encoder = LabelEncoder()
    # label_encoder = label_encoder.fit(Y)
    # label_encoded_y = label_encoder.transform(Y)


def compare_submissions(sub1, sub2, dir_path_sub, feature_name='y'):
    pred_1 = pd.read_csv(dir_path_sub + sub1)
    pred_2 = pd.read_csv(dir_path_sub + sub2)
    show_ks_stat(pred_1, pred_2, feature_name)
    print(pred_1[feature_name].min(), pred_1[feature_name].mean(),
          pred_1[feature_name].max(), pred_1[feature_name].std(),
          ' for ', sub1)

    print(pred_2[feature_name].min(), pred_2[feature_name].mean(),
          pred_2[feature_name].max(), pred_2[feature_name].std(),
          ' for ', sub2)


def find_low_variance(df, max_v=0.0001, is_print=True):
    res = set()
    for k, v in df.var().items():
        if v <= max_v:
            if is_print:
                print(k, v)
            res.add(k)

    return res


def find_high_variance(df, min_v=.002, is_print=True, return_scores=False):
    res = list()
    res2 = {}
    for k, v in df.var().items():
        if v >= min_v:
            if is_print:
                print(k, v)
            res.append(k)
            if return_scores:
                res2.update({k: v})

    return res2 if return_scores else res


def find_significant_diff_variance(df1, df2, is_print=True):
    res = list()
    min_expected_variance_diff = 1

    var2 = df2.var().items()

    for k, v in df1.var().items():
        if abs(v - var2[k]) / (v + 0.0000001) >  min_expected_variance_diff:
            if is_print:
                print(k, v, var2[k])
            res.append(k)

    return res


def show_heatmap(data):
    sns.heatmap(data.corr())
    plt.show()

