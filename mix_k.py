import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.preprocessing as preprocessing
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, median_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from kaggle.sbank.versions.clean_sb_data import *

warnings.filterwarnings('ignore')

# - исследовать два небольших пика слева на графике цен
# - цена по районам
# - использовать период постройки вместо года
# - цена на первом этаже для магазина
# - если первый этаж и много населения в районе
# - при росте стоимости кредита цена новостроя падает


# Цены на квартиры в Москве, рублей за кв.м. (www.metrinfo.ru)	22.05.17	к 15.05.17
# Старая панель (5-этажки и иные квартиры с маленькой кухней)	137 070	+0,3%
# Типовая панель (9-14 этажей, типовые площади)	142 957	+0,3%
# Современная панель (от 16 эт. и иные кв. увеличенных пл-дей)	155 247	-0,3%
# Старый кирпич (5-этажки и иные квартиры с маленькой кухней)	156 390	+0,2%
# Сталинки и типовой кирпич (6-11 эт., и иные кв. небол. пл-дей)	190 343	-0,2%
# Современный монолит-кирпич (монолиты, кирпич увел. пл-дей)	216 465	0,0%
# Все панельные и блочные дома	145 091	+0,1%
# Все монолитные и кирпичные дома	187 733	0,0%


# рядом метро - фактор
# num_room - возможно квартира-студия, стоит дешевле. В ней может кухня 2 кв.м
# num_room может быть не указан, если в квартире ще нет отделки
# рассмотреть 5,9 - этажки
# втор:новострой 3:1
# !is_na_nearest_metro
# выбросить склады с большой площадью ( > 1000 )

# KITCHEN INCLUDED IN LIFE SQ CHECK INCONSISTENCY (DONE)\n",
#         "* FULL SQ > LIFE SQ (MOST PROBABLY) (DONE)\n",
# https://www.kaggle.com/keremt/very-extensive-cleaning-by-sberbank-discussions/code


# https://www.kaggle.com/rareitmeyer/av-explained-and-exploited-for-better-prediction/comments
# https: // www.kaggle.com / c / sberbank - russian - housing - market / discussion / 33272
# https://www.kaggle.com/c/sberbank-russian-housing-market/discussion/32717
# https://www.kaggle.com/agzamovr/a-very-extensive-exploratory-analysis-in-python/comments
# https://www.kaggle.com/wti200/deep-neural-network-for-starters-r/code#L73
# https://www.kaggle.com/aharless/kernels

# !! http://www.xavierdupre.fr/app/pymyinstall/helpsphinx/notebooks/example_xgboost.html
# http://programtalk.com/python-examples/xgboost.XGBClassifier/
# Simple: http://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/

# http://dsdeepdive.blogspot.com/2015/06/linear-regression-with-python.html
# https://www.kaggle.com/c/sberbank-russian-housing-market/discussion/32533
# https://www.kaggle.com/c/sberbank-russian-housing-market/discussion/32481
# https://www.kaggle.com/ffisegydd/sklearn-multicollinearity-class/comments/comments
# https: // www.kaggle.com / c / sberbank - russian - housing - market / discussion / 32436
# https://www.kaggle.com/lokendradevangan/housing-price-feature-selection/comments/comments
# https://www.kaggle.com/aharless/exploration-of-sberbank-housing-data-part-i/comments
# https://www.kaggle.com/abhishekkant/another-xgb-model/comments/comments
# https://www.kaggle.com/c/sberbank-russian-housing-market/discussion/32222
# https://www.kaggle.com/georsara1/my-first-try-on-russian-house-market-prices/comments/comments
# https: // www.kaggle.com / c / sberbank - russian - housing - market / discussion / 32408
# https://www.kaggle.com/c/sberbank-russian-housing-market/discussion/32406
# https://www.kaggle.com/c/sberbank-russian-housing-market/discussion/32391
# https://www.kaggle.com/c/sberbank-russian-housing-market/discussion/32303
# https://www.kaggle.com/c/sberbank-russian-housing-market/discussion/32328
# https://www.kaggle.com/prokopyev/naive-xgb/comments/comments
# https://www.kaggle.com/c/sberbank-russian-housing-market/discussion/32567
# https://www.kaggle.com/c/sberbank-russian-housing-market/discussion/32312

# ? https://www.kaggle.com/rdslater
# о валидации
# https://www.kaggle.com/aharless/weighted-least-squares-sberbank-data
# https://www.kaggle.com/aharless/exploration-of-sberbank-housing-data-part-i
# https://www.kaggle.com/aharless/sloppy-mess-ols
# https://www.kaggle.com/aharless/probabilistic-version-of-small-improvements
# https://www.kaggle.com/aharless/estimating-1m-2m-3m-incidence-over-time
# https://www.kaggle.com/aharless/from-vignesh-mohandas-xgboost-model-3
# https://www.kaggle.com/aharless/jiwon-s-small-improvements
# https://www.kaggle.com/aharless/why-i-gave-up-on-heteroskedasticity-for-now
# https://www.kaggle.com/aharless/using-nonlinear-prediction-algorithms
# https://www.kaggle.com/aharless/automatic-linear-model-weighting-and-svm
# https://www.kaggle.com/aharless/linear-models-for-house-prices
# https://www.kaggle.com/aharless/from-bruno-do-amaral-naive-xgb-v9
# https://www.kaggle.com/aharless/sberbank-data-wls-with-heteroskedasticity-weights
# https://www.kaggle.com/aharless/an-interesting-fact-about-investment-properties


# params = {}
# params['objective'] = 'reg:linear'
# params['eta'] = 0.05
# params['max_depth'] = 6
# params['subsample'] = 0.9
# params['colsample_bytree'] = 0.7
# params['min_child_weight'] = 1.0
# params['gamma'] = 0.05
# params['reg_lambda'] = 1.0
# params['silent'] = 1
#
# watchlist = [(d_train, 'train'), (d_valid, 'valid')]
# clf = xgb.train(params, d_train, 600, watchlist, feval=rmsle, verbose_eval=10, early_stopping_rounds=100)

# Идея исправлять пропуски с k-nearest
# Идея использовать внешний источник данных по безработным

dir_path = os.path.dirname(os.path.realpath(__file__))
fields_test = [
    # 'timestamp', 'full_sq', 'num_room', 'build_year', 'state',
    #            'workplaces_km', 'floor', 'max_floor',
    #            'material', 'sub_area', 'life_sq', 'kremlin_km', 'mkad_km',
    'product_type',

    'ecology',
    'id', 'timestamp', 'full_sq', 'life_sq', 'floor', 'max_floor', 'material', 'build_year', 'num_room',
    'kitch_sq', 'state', 'sub_area', 'area_m', 'green_zone_part',
    'preschool_quota',
    # 'raion_build_count_with_material_info',
    # 'ID_metro',
    'metro_min_avto',
    # 'water_treatment_km',
    'school_km', 'park_km', 'industrial_km', 'incineration_km',
    'public_transport_station_min_walk',  'mkad_km', 'sadovoe_km', 'kremlin_km',
    'railroad_km', 'zd_vokzaly_avto_km', 'bus_terminal_avto_km', 'oil_chemistry_km',
    'big_market_km', 'market_shop_km', 'fitness_km',
    'detention_facility_km', 'workplaces_km', 'shopping_centers_km', 'office_km',
    'preschool_km', 'trc_sqm_5000', 'prom_part_5000', 'office_sqm_5000',
    'green_part_5000', 'radiation_km', 'big_road1_km', 'big_road2_km',
    'power_transmission_line_km', 'thermal_power_plant_km', 'radiation_raion',
    # 'incineration_raion', 'railroad_terminal_raion', 'railroad_1line',
    # 'nuclear_reactor_raion',
    # 'cafe_count_5000',

    # 'kindergarten_km',
    #?  'water_km', 'indust_part','cemetery_km',

    # 'sport_objects_raion',
    # 'additional_education_raion',
    # 'build_count_brick', 'build_count_monolith', 'build_count_panel',
    'raion_build_count_with_builddate_info',
    'build_count_before_1920', 'build_count_1921-1945', 'build_count_1946-1970',
    'build_count_1971-1995', 'build_count_after_1995',
    'green_zone_km',
]

# fields_my_test = ['full_sq', 'num_rooms', 'min_walk_m', 'floor', 'max_floor']
my_fields = ['full_sq', 'num_rooms', 'min_walk_m', 'floor', 'max_floor']

fields = ['price_doc'] + list(fields_test)

# fields_test

# So the top 5 variables and their description from the data dictionary are:\n",
# " 1. full_sq - total area in square meters, including loggias, balconies and other non-residential areas\n",
# " 2. life_sq - living area in square meters, excluding loggias, balconies and other non-residential areas\n",
# " 3. floor - for apartments, floor of the building\n",
# " 4. max_floor - number of floors in the building\n",
# " 5. build_year - year built\n",

# 'metro_min_walk',
# use for setting macro environments 'timestamp'
# use material
# 'max_floor',

# 1 - panel, 2 - brick, 3 - wood, 4 - mass concrete, 5 - breezeblock, 6 - mass concrete plus brick

# merge with radio factors 'oil_chemistry_km',
# build_year - group by periods: ant, sov, new ...

fields_macro = ['timestamp', 'brent', 'usdrub', 'eurrub', 'ppi', 'fixed_basket',
                # TODO check 'rent_price_4+room_bus'
                # 'micex_cbi_tr', 'mortgage_rate',
                ]

f_t = ['id'] + fields_test


test2 = pd.read_csv(dir_path + '/my_test.csv', header=0)

# print(test2)
# combine_factors(test2, ['full_sq', 'num_rooms', 'floor', 'max_floor'])
# s()


# TODO merge macro
macro = pd.read_csv(dir_path + '/macro.csv', header=0, usecols=fields_macro)
train = pd.read_csv(dir_path + '/train.csv', header=0, usecols=fields)
test = pd.read_csv(dir_path + '/test.csv', header=0, usecols=f_t)

set_year(train)
set_year(test)

# visualize_feature_over_time(macro, 'fixed_basket')
# print(macro[['fixed_basket']].tail())

# n = train[np.isnan(train['full_sq'])
          # np.isnan(train['life_sq']) &
          # np.isnan(train['floor']) &
          # np.isnan(train['max_floor'])
          # np.isnan(train['material']) &
          # np.isnan(train['build_year']) &
          # np.isnan(train['num_room']) &
          # np.isnan(train['kitch_sq']) &
          # np.isnan(train['state'])
          # np.isnan(train['product_type']) &
          # np.isnan(train['sub_area'])
          # ]

# print(n)

train = pd.merge(macro, train, how='right', on=['timestamp'])
test = pd.merge(macro, test, how='right', on=['timestamp'])


def an_xgb(file_suffix, post=None):
    # Basic XBoost model
    # https://github.com/dmlc/xgboost/blob/master/demo/guide-python/generalized_linear_model.py
    for f in train.columns:
        if train[f].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train[f].values))
            train[f] = lbl.transform(list(train[f].values))

    train_y = train.price_doc.values
    if post:
        train_X = train.drop(["price_doc"], axis=1)
    else:
        train_X = train.drop(['id', "timestamp", "price_doc"], axis=1)

    xgb_params = {
        'eta': 0.05,
        'max_depth': 8,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1
    }

    dtrain = xgb.DMatrix(train_X, train_y, feature_names=train_X.columns.values)
    model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100)

    # plot the important features #
    fig, ax = plt.subplots(figsize=(12, 18))
    xgb.plot_importance(model, height=0.8, ax=ax)
    plt.savefig(os.path.basename(__file__) + '200_clean' + file_suffix + '.png')

    # https: // www.dataiku.com / learn / guide / code / python / advanced - xgboost - tuning.html

    # plt.show()

info_fields = [
    'price_doc', 'pbm_usd', 'build_year', 'full_sq', 'life_sq',
    'num_room', 'sub_area_origin', 'product_type'
]


def format_data(raw_data, is_production=None):
    raw_data['product_type'] = raw_data['product_type'].map({'Investment': 0, 'OwnerOccupier': 1})
    # raw_data.loc[np.isnan(raw_data['product_type']) & np.isnan(raw_data['build_year'])] = 0
    # print(
        # raw_data.loc[
        #     np.isnan(raw_data['build_year'])
        #     & ~np.isnan(raw_data['life_sq'])
        #     & (raw_data['timestamp'] > '2014-07-25')
        #
        # ]
        # raw_data.groupby('product_type', as_index=False)['product_type'].agg(['count'])
    # )
    # asd()
    # For test data
    # All ( 5604 : 1 ) with Nan build_year & Nan life_sq are Owners - 1
    raw_data.loc[np.isnan(raw_data['build_year']) &
                 # np.isnan(raw_data['life_sq']) &
                 np.isnan(raw_data['product_type'])
                 , 'product_type'] = 1

    # asd()

    set_av(raw_data, 'product_type')
    # raw_data['product_type'] = raw_data['product_type'].astype(int)

    # mean_life_to_full_square = np.mean(raw_data['life_sq']) / np.mean(raw_data['full_sq'])
    # print(mean_life_to_full_square)
    # percentile_validate(raw_data, 'life_sq')

    # Non-residential area to living area ratio (full_sq - life_sq) / full_sq
    create_bi_val(raw_data)

    if is_production is None:
        # print(len(raw_data[train['full_sq'] == 0]))
        raw_data = raw_data[~np.isnan(train['full_sq']) & (train['full_sq'] > 0)]

        llimit = np.percentile(train['price_doc'].values, 0.05)
        rlimit = np.percentile(train['price_doc'].values, 99.95)
        # raw_data['is_prise_more'] = (raw_data['price_doc'] > rlimit)
        # raw_data['is_prise_less'] = (raw_data['price_doc'] < llimit)
        # raw_data = raw_data[train['price_doc'] > llimit]
        # raw_data.loc[raw_data['price_doc'] > rlimit, 'price_doc'] = rlimit
        raw_data = raw_data[(raw_data['price_doc'] > llimit)
                            & (raw_data['price_doc'] < rlimit)]

        filter_full_sq(raw_data, is_production)

        # Fix missed decimal point
        raw_data.loc[raw_data['life_sq'] > 300, 'life_sq'] = (raw_data['life_sq'] / 10)
        raw_data.loc[raw_data['full_sq'] < raw_data['life_sq'], 'life_sq'] = raw_data['life_sq'] / 10

        raw_data = raw_data[train['max_floor'] < 50]

        # llimit = np.percentile(train['price_doc'].values, 0.05)
        # rlimit = np.percentile(train['price_doc'].values, 99.95)
        # raw_data['is_prise_more'] = (raw_data['price_doc'] > rlimit)
        # raw_data['is_prise_less'] = (raw_data['price_doc'] < llimit)
        # raw_data = raw_data[train['price_doc'] > llimit]
        # raw_data.loc[raw_data['price_doc'] > rlimit, 'price_doc'] = rlimit

        # TODO validate full_sq
        raw_data['pbm'] = (raw_data['price_doc'] / raw_data['full_sq'])
        raw_data['pbm_usd'] = raw_data['pbm'] / raw_data['usdrub']
        raw_data['pbm'] = raw_data['pbm'].astype(int)
        raw_data['pbm_usd'] = raw_data['pbm_usd'].astype(int)

        # print(raw_data.groupby('max_floor', as_index=False)
        #     ['pbm_usd', 'build_year'].agg(['min', 'mean', 'max', 'count']))

        # f = raw_data[raw_data['max_floor'] == 0][['price_doc']]
        # l = raw_data[raw_data['max_floor'] == 3][['price_doc']]

        # print(ks_2samp(f.values[:, 0], l.values[:, 0]))
        #
        # asd()
        # np.isnan(train['life_sq']) &
        # np.isnan(train['floor']) &
        # np.isnan(train['max_floor'])
        # np.isnan(train['material']) &
        # np.isnan(train['build_year']) &
        # np.isnan(train['num_room']) &

        # train = train.loc[train['full_sq'] > 3]
        # train = train.loc[~((train['price_doc'] == 1e6) & (train['pbm_usd'] < 200))]
        # train = train.loc[~((train['price_doc'] == 2e6) & (train['pbm_usd'] < 400))]
        # train = train.loc[~((train['price_doc'] == 3e6) & (train['pbm_usd'] < 500))]

        # raw_data['LogAmt'] = np.log(raw_data.price_doc + 1.0)
        # describe(raw_data, 'LogAmt')
        # sns.distplot(raw_data['LogAmt'])
        # plt.show()

        # train['price_doc'] /= train['cpi']

        # TODO Train another classifier for records with price more then rlimit
        # and use it for prediction of test['price_doc'] > rlimit
        # train['price_doc'] = np.log1p(train['price_doc'].values)

        # Сезонность
        # visualize_feature_over_time(train, 'price_doc')
        # visualize_feature_over_time(train, 'usdrub')
        # describe(train, 'mu')
        # get_correlation(train, 'fixed_basket', 'price_doc')
        # show_pair_plot(train, ['mu', 'cpi', 'usdrub', 'price_doc'])
        # combine_factors(train, ['usdrub', 'ppi', 'fixed_basket'])
    else:
        raw_data.loc[raw_data['max_floor'] >= 50, 'max_floor'] = np.NAN
        set_av(raw_data, 'max_floor')
        filter_full_sq(raw_data, is_production)
        # Fix missed decimal point
        raw_data.loc[raw_data['life_sq'] > 300, 'life_sq'] = (raw_data['life_sq'] / 10)
        raw_data.loc[raw_data['full_sq'] < raw_data['life_sq'], 'life_sq'] = raw_data['life_sq'] / 10

    raw_data['bi_val'] **= -1

    t = raw_data.loc[raw_data['sub_area'] == 'Chertanovo Juzhnoe']
    print(t['pbm_usd'])
    show_hist(t['pbm_usd'])
    print(t.groupby('pbm_usd', as_index=False)
        ['pbm_usd'].agg(['min', 'mean', 'max', 'count']))

    print()
    asd()

    # raw_data.insert(0, 'life_sq_pers', np.NAN)
    # raw_data['life_sq_pers'] = (raw_data['full_sq'] - raw_data['life_sq']) / raw_data['full_sq']
    # raw_data.loc[np.isnan(raw_data['life_sq_pers']), 'life_sq_pers'] = 0

    # print(train[train['price_doc'] < llimit][['price_doc', 'sub_area', 'life_sq',
    #                                           'full_sq', 'kremlin_km']])
    # train = train[train['price_doc'] < rlimit]
    # sys.exit(0)

    # an_xgb('_all_')

    # show_missing_values(test)
    # asd()
    raw_data['sub_area_origin'] = raw_data['sub_area']

    # handle life_sq = 1 in both datasets
    # print(test[(test['full_sq'] < 3)][info_fields])

    # print(train[(train['kremlin_km'] < 0.8)
    #             & (train['pbm_usd'] < 4000)][['price_doc',
    #                                           'pbm_usd', 'build_year',
    #                                           'full_sq', 'num_room',
    #                                           'sub_area_origin']])

    # describe(test, 'kremlin_km')

    # describe(raw_data, 'full_sq', nlo=40)
    # asd()
    # show_correlations(raw_data, 'life_sq_pers')
    # asd()
    # describe(raw_data, 'life_sq_pers')

    # raw_data = ['test_col'] + raw_data
    # print(raw_data)
    # asd()

    # raw_data.insert(0, 'floor_to', np.NAN)
    # raw_data.loc[
    #     ~np.isnan(raw_data['max_floor'])
    #     & (raw_data['max_floor'] > 0)
    #     & ~np.isnan(raw_data['floor'])
    #     & (raw_data['max_floor'] >= raw_data['floor']),
    #     'floor_to'] = raw_data['floor'] / raw_data['max_floor']
    # raw_data.loc[np.isnan(raw_data['floor_to']), 'floor_to'] = np.mean(raw_data['floor_to'])

    # show_correlations(raw_data, 'floor_to')
    # show_pair_plot(raw_data, ['floor_to', 'price_doc'])
    # asd()
    # describe(raw_data, 'floor_to')
    # asd()

    filter_kremlin_km(raw_data, is_production)
    percentile_validate(raw_data, 'metro_min_avto')
    filter_subarea(raw_data)
    filter_ecology(raw_data)

    # default_validate(raw_data, 'workplaces_km', 0, 50)
    # default_validate(raw_data, 'metro_min_walk', 0, 1000)
    # raw_data['park_km'] = np.log(raw_data['park_km'])
    # raw_data['park_km'] = raw_data['park_km'].astype(int)
    # describe(raw_data, 'park_km')
    # show_correlations(raw_data, 'park_km')
    # asd()

    # describe(raw_data, 'trc_sqm_5000')
    # asd()

    percentile_validate(raw_data, 'park_km')
    raw_data['park_km'] = np.log(raw_data['park_km'])
    # show_correlations(raw_data, 'park_km')
    # asd()

    percentile_validate(raw_data, 'school_km')

    # print(raw_data[np.isnan(raw_data['prom_part_5000'])])
    # show_correlations(raw_data, 'prom_part_5000')
    # asd()

    # 'mkad_km', 'water_1line'
    # raw_data['full_sq'] = raw_data['full_sq'] ** 1.45
    # describe(raw_data, 'full_sq', nlo=30)
    # visualize_feature_over_time()
    # show_bar_plot(raw_data, 'full_sq', 'price_doc')
    # get_correlation(raw_data, 'full_sq', 'price_doc')
    # get_correlation(raw_data, 'full_sq_log_2', 'price_doc')
    # get_correlation(raw_data, 'full_sq_log', 'price_doc')
    # get_correlation(raw_data, 'full_sq_log_e', 'price_doc')

    # describe(raw_data, 'metro_min_avto')
    # get_correlation(raw_data, 'metro_min_avto', 'price_doc')
    # asd()

    # visualize_feature_over_time(train, 'pbm_usd')
    raw_data['is_na_floor'] = np.isnan(raw_data['floor'])
    raw_data['is_zero_floor'] = (raw_data['floor'] == 0)
    raw_data['is_first_floor'] = (raw_data['floor'] == 1)

    set_av(raw_data, 'floor')
    percentile_validate(raw_data, 'floor')

    raw_data.loc[raw_data['state'] > 4, 'state'] = np.NAN
    default_validate(raw_data, 'state', 0, 4, 0)
    raw_data.loc[raw_data['material'] == 3, 'material'] = np.NAN
    default_validate(raw_data, 'material', 1, 6)

    # show_correlations(raw_data, 'state')
    #
    raw_data['state'] **= 2

    # raw_data['product_type'] = raw_data['product_type'].map({'Investment': 0, 'OwnerOccupier': 1})
    # raw_data.loc[np.isnan(raw_data['product_type']) & np.isnan(raw_data['build_year'])] = 0
    # set_av(raw_data, 'product_type')
    # raw_data['product_type'] = raw_data['product_type'].astype(int)

    # raw_data.loc[
    #     np.isnan(raw_data['product_type']) & raw_data['price_doc'] % 1e3,
    #     'product_type'] = 0
    # set_av(raw_data, 'product_type')

    # ~(raw_data['product_type'] >= 0)

    filter_build_year(raw_data)

    # y3 = exp((log(y1) + log(y2)) / 2)

    # describe(train, 'floor')
    # visualize_feature_over_time(train, 'is_na_floor')

    # get_correlation(train, 'is_first_floor', 'pbm')
    # show_pair_plot(train, ['floor', 'price_doc'])
    # combine_factors(train, ['floor', 'pbm_usd'])

    # print(train[np.isnan(train['floor'])]['price_doc'].mean())

    # s()
    # print(train[train['floor'] < 1]['pbm_usd'].mean())
    # print(train[train['floor'] == 1]['pbm_usd'].mean())
    # print(train[train['floor'] > 1]['pbm_usd'].mean())
    # print(train[train['floor'] == 2]['pbm_usd'].mean())
    # print(train[train['floor'] == 3]['pbm_usd'].mean())
    # print(train[train['floor'] == 4]['pbm_usd'].mean())
    # print(train[train['floor'] == 5]['pbm_usd'].mean())
    # print(train[train['floor'] == 6]['pbm_usd'].mean())

    # describe(train, 'floor')
    # train['is_first_floor'] = (train['floor'] == 1)

    # show_correlations(train, 'floor', 'price_doc')

    # visualize_feature_over_time(train, 'pbm_usd')
    # visualize_feature_over_time(train, 'floor')
    # show_bar_plot(train, 'floor', 'sale_year')

    # print(train[train['floor'] == 0])
    # print(train[np.isnan(train['floor'])])

    # visualize full_sq groups
    # default_validate(raw_data, 'full_sq', 10, 150)
    # percentile_validate(raw_data, 'life_sq', skip_left_outlines=1)

    rlimit = np.percentile(raw_data['mkad_km'].values, 99.5)
    llimit = np.percentile(raw_data['mkad_km'].values, 0.5)
    raw_data.loc[raw_data['mkad_km'] > rlimit, 'mkad_km'] = np.mean(raw_data['mkad_km'])
    raw_data.loc[raw_data['mkad_km'] < llimit, 'mkad_km'] = llimit

    # print(raw_data['center_km'])
    # 'oil_chemistry_km', 'thermal_power_plant_km'

    # raw_data['oil_chemistry_km'] /= 2
    # raw_data['oil_chemistry_km'] = raw_data['oil_chemistry_km'].astype('int')
    # raw_data['oil_chemistry_km'] = raw_data['oil_chemistry_km'] < 3

    # sns.pairplot(raw_data[['kremlin_km', 'oil_chemistry_km', 'price_doc']], kind="reg")
    # plt.show()
    # sys.exit()
    #
    # plt.hist(train['mkad_km'], bins=10)
    # plt.show()

    # sys.exit(0)

    # default_validate(raw_data, 'max_floor', 1, 40)

    # rlimit = np.percentile(train['max_floor'].values, 99.5)
    # llimit = np.percentile(train['max_floor'].values, 0.5)
    # print(rlimit, llimit)
    # print(train[train['floor'] > train['max_floor']])
    # print(train[train['max_floor'] == 0])
    # sns.barplot(x="max_floor", y="price_doc", data=train)
    # plt.show()

    # raw_data['max_floor'] = raw_data['max_floor'].astype(int)
    # raw_data['floor'] = raw_data['floor'].astype(int)
    # raw_data['last_floor'] = raw_data.loc[raw_data['floor']
    #                                    & raw_data['max_floor']
    #                                    & (raw_data['floor'] == raw_data['max_floor'])]
    #
    # print(stats.pearsonr(raw_data[['last_floor']], raw_data[['price_doc']]))

    # sns.barplot(x="last_floor", y="price_doc", data=raw_data)
    # plt.show()
    # plt.hist(train['max_floor'], bins='auto')
    # plt.show()

    rlimit = np.percentile(raw_data['floor'].values, 99.5)
    llimit = np.percentile(raw_data['floor'].values, 0.5)
    default_validate(raw_data, 'floor', rlimit, llimit)

    percentile_validate(raw_data, 'kitch_sq', custom_default=0)
    # percentile_validate(raw_data, 'kindergarten_km')

    raw_data.loc[np.isnan(raw_data['prom_part_5000']), 'prom_part_5000'] \
        = np.mean(raw_data['prom_part_5000'])

    # percentile_validate(raw_data['prom_part_5000'])
    # percentile_validate(raw_data, 'office_sqm_5000', skip_left_outlines=1)
    # percentile_validate(raw_data, 'indust_part')
    # percentile_validate(raw_data, 'cemetery_km')
    # percentile_validate(raw_data, 'water_km')

    # default_validate(raw_data, 'num_room', 1, 15)

    rlimit = np.percentile(raw_data['num_room'].values, 99.5)
    default_validate(raw_data, 'num_room', 1, rlimit)

    # print(stats.pearsonr(train[['full_sq']], train[['price_doc']]))
    # print(stats.pearsonr(train[['num_room']], train[['price_doc']]))
    # print(stats.pearsonr(train[['num_room']], train[['full_sq']]))
    # sns.pairplot(raw_data[['full_sq', 'num_room', 'price_doc']], kind="reg")
    # plt.show()
    # s()

    # asd()

    # raw_data['missing_state'] = ~((raw_data['state'] > 0) & (raw_data['state'] <= 4))

    # sns.pairplot(raw_data[['state', 'price_doc', 'brent', 'missing_state']], kind="reg")
    # plt.show()
    # sys.exit(0)

    # TODO discover first and last floor
    # raw_data = raw_data[(raw_data['max_floor'] > 0) & (raw_data['floor'] <= raw_data['max_floor'])]
    # raw_data['from_max_floor'] = \
    #     (raw_data['floor'] == 1) or (raw_data['max_floor'] == raw_data['floor'])

    # sns.pairplot(raw_data[['sub_area_cat', 'price_doc', 'state']], kind="reg")
    # plt.show()

    # raw_data['product_type'] = raw_data['product_type'].fillna('Investment')
    # default_validate(raw_data, 'floor', 0, 35)

    # raw_data['usdrub'] = np.log(raw_data['usdrub'])
    # show_correlations(raw_data, 'usdrub', 'price_doc')
    # asd()

    # print(len(raw_data[
    # np.isnan(raw_data['product_type']) & (raw_data['price_doc'] % 1e4 == 0)
    #       ]['product_type']))

    # set_av(raw_data, 'product_type')
    # raw_data['product_type'] = raw_data['product_type'].astype(int)

    # Кандидаты на удаление: floor, material, num_room
    raw_data = raw_data.drop([
                              'workplaces_km',
                              # TODO delete
                              # 'sub_area_origin',
                              'brent',
                              'num_room',
                              'material',
                             'life_sq',

                              # For ecology prediction
                              'green_zone_km', 'oil_chemistry_km',
                              'big_road1_km', 'big_road2_km', 'radiation_km',
                              'radiation_raion', 'green_part_5000',
                              'power_transmission_line_km', 'thermal_power_plant_km',

                              # For build year prediction
                              'raion_build_count_with_builddate_info',
                              ], axis=1)

    # check_mean(raw_data, 'metro_min_walk')
    # plt.hist(train['metro_min_walk'], bins=10)
    # plt.show()
    # sys.exit(0)

    return raw_data


# Now let us see how these important variables are distributed with respect to target variable
# ulimit = np.percentile(train['price_doc'].values, 99.5)
# llimit = np.percentile(train['price_doc'].values, 0.5)
# train['price_doc'].ix[train['price_doc'] > ulimit] = ulimit
# train['price_doc'].ix[train['price_doc'] < llimit] = llimit
#
# train[col].ix[train[col] > ulimit] = ulimit
# train[col].ix[train[col] < llimit] = llimit
#
# plt.figure(figsize=(12, 12))
# sns.jointplot(x=np.log1p(train['full_sq'].values), y=np.log1p(train['price_doc'].values), size=10)
# plt.ylabel('Log of Price', fontsize=12)
# plt.xlabel('Log of Total area in square metre', fontsize=12)
# plt.show()
# sys.exit()

# train = train[~(np.isnan(train['num_room']) & np.isnan(train['build_year']))]
# t = train[~np.isnan(train['full_sq']) & train['num_room'] > 0]
# t['sq_per_room'] = t['full_sq'] / t['num_room']
# avg_room_size = t['sq_per_room'].mean()

# print(len(train[np.isnan(train['full_sq'])]))

# train[~np.isnan(train['full_sq']) & np.isnan(train['num_room'])]['num_room'] = \
#     round(train['full_sq'] / avg_room_size)

# print(train[['timestamp', 'floor', 'full_sq', 'num_room', 'state',
#             'build_year', 'product_type', 'price_doc']].head(7000))

# print(train[~np.isnan(train['full_sq']) & np.isnan(train['num_room']) & ~np.isnan(train['build_year'])])
# print(train[['timestamp', 'floor', 'full_sq', 'num_room', 'sq_per_room', 'state',
#             'build_year', 'product_type', 'price_doc']].head(7000))

# macro.loc[macro['timestamp'] == '2010-01-03', 'usdrub'] = 29.9
# print(train.info())

# print(train[['deposits_value', 'deposits_growth', 'deposits_rate',]])
# train.loc[np.isnan(train['metro_min_walk']), 'metro_min_walk'] = np.mean(train['metro_min_walk'])
# ['metro_min_avto', 'metro_km_avto', 'metro_min_walk', 'price_doc']])

# full_data = [test]
# check(train, 'state')

# check_nan(train, 'product_type')
# raw_data['product_type'] = raw_data['product_type'] \
#     .fillna(raw_data['product_type'].mean())

# 'usdrub', 'full_sq', 'state',
# 'kindergarten_km', 'school_km', 'park_km', 'industrial_km', 'sport_count_5000'

# train['kindergarten_km'] *= 1000
# train['kindergarten_km'] = train['kindergarten_km'].astype('int')
# rlimit = np.percentile(train['kindergarten_km'].values, 99.5)
# llimit = np.percentile(train['kindergarten_km'].values, 0.5)
# train.loc[train['kindergarten_km'] > rlimit, 'kindergarten_km'] = np.NAN
# set_av(train, 'kindergarten_km')
#
# print(rlimit, llimit)
# check_outlines(train, 'price_doc', show_left=1)
# print(stats.pearsonr(train[['sport_count_5000']], train[['price_doc']]))
# print(stats.pearsonr(train[['sport_count_5000']], train[['school_km']]))
# g = sns.pairplot(train[['sport_count_5000', 'school_km', 'price_doc']], kind="reg")
# plt.show()

# First glance at our var
# plt.figure(figsize=(8, 6)),
# plt.scatter(range(train.shape[0]), np.sort(train.price_doc.values)),
# plt.xlabel('index', fontsize=12),
# plt.ylabel('price', fontsize=12),
# plt.show()

# Distribution
# plt.figure(figsize=(12, 8))
# sns.distplot(train.price_doc.values, bins=50, kde=True)
# plt.xlabel('price', fontsize=12)
# plt.show()

# Log distribution
# plt.figure(figsize=(12, 8))
# sns.distplot(np.log(train.price_doc.values), bins=50, kde=True)
# plt.xlabel('price', fontsize=12)
# plt.show()

# train_df['yearmonth'] = train_df['timestamp'].apply(lambda x: x[:4]+x[5:7])\
# grouped_df = train_df.groupby('yearmonth')['price_doc'].aggregate(np.median).reset_index()

# train['year_month'] = train['timestamp'].dt.year*100 + train["timestamp"].dt.month
# train = train.drop(['id', 'timestamp'], axis=1)
# Adding log price for use as target variable
# train['log_price_doc'] = np.log1p(train['price_doc'].values)

# df_na_row = train.dropna(axis=0)
# train.dropna(axis=1)
# print(len(df_na_row))


# print(len(train))
# asd()

# train = train.drop(['life_sq', 'micex_cbi_tr', 'kitch_sq',
#                     'ID_metro', 'brent'], axis=1)

# an_xgb('_filter_nan_new_')
# exit(0)

# percentile_validate(train, 'park_km')
# percentile_validate(train, 'industrial_km')

# print(stats.pearsonr(train[['price_doc']], train[['raion_popul']]))
# percentile_validate(train, 'raion_build_count_with_material_info', custom_default=0)
# percentile_validate(train, 'railroad_station_walk_km')
# train['public_transport_station_min_walk'] = train['public_transport_station_min_walk'].astype(int)

# ax = sns.barplot(x="build_year", y="price_doc", data=train)
# sns.pairplot(train[['metro_min_avto', 'price_doc']], kind="reg")

# an_xgb('after_clean_nl')
# sys.exit(0)

# Convert to USD
# train['price_origin'] = train['price_doc']
# train['price_doc'] /= train['usdrub']
# train['price_doc'] = train['price_doc'].astype(int)
# train = train.drop(['usdrub'], axis=1)

# TODO review bottom price
# First
# train = train[train['price_doc'] > 10000]
# train = train[train['price_doc'] < 1400000]
# Stable
# rlimit = np.percentile(train['price_doc'].values, 99.5)
# llimit = np.percentile(train['price_doc'].values, 0.5)
# train.loc[train['price_doc'] > rlimit, 'price_doc'] = rlimit
# train = train[train['price_doc'] > llimit]
# train = train[train['price_doc'] < rlimit]
# train['price_doc'] = train['price_doc'].astype(int)
# train = train[train['price_doc'] > 450000]
# train = train[train['price_doc'] < 80000000]

# plt.hist(train['price_doc'], bins='auto')
# plt.show()

# print(train[train.product_type == "Investment"].price_doc.value_counts().head(100))
# asd()

# train = train.loc[~(train['pbm_usd'] < 200)]
# train = train.loc[~((train['price_doc'] == 1e6) & (train['pbm_usd'] < 300))]
# train = train.loc[~((train['price_doc'] == 2e6) & (train['pbm_usd'] < 400))]
# train = train.loc[~((train['price_doc'] == 3e6) & (train['pbm_usd'] < 500))]

# print(len(train.loc[(train['pbm_usd'] < 0)]))
# print(len(train.loc[(train['pbm_usd'] < 100)]))
# print(len(train.loc[(train['pbm_usd'] < 500)]))
# print(len(train.loc[(train['pbm_usd'] < 1000)]))
# print(len(train.loc[(train['pbm_usd'] < 2000)]))
# print(len(train.loc[(train['pbm_usd'] < 5000)]))
# print(len(train.loc[(train['pbm_usd'] < 10000)]))
# print(len(train.loc[(train['pbm_usd'] > 10000)]))
# print(train.loc[(train['price_doc'] == 1e6) & (train['pbm_usd'] > 500)])
# print(len(train.loc[train['price_doc'] == 2e6]))
# print(len(train.loc[train['price_doc'] == 3e6]))
# print(len(train.loc[train['price_doc'] == 4e6]))

# asd()


train = format_data(train)

#dropf
train = train.drop(['timestamp', 'id',
                    # 'sub_area',
                    # 'build_year',
                    'usdrub', 'eurrub',

                    # 'deposits_value',
                    # 'state',
                    'green_zone_part',
                    'preschool_quota', 'incineration_km', 'mkad_km', 'sadovoe_km',
                    'railroad_km', 'zd_vokzaly_avto_km', 'bus_terminal_avto_km',
                    'big_market_km', 'market_shop_km', 'fitness_km',
                    'detention_facility_km', 'shopping_centers_km',
                    'office_km', 'preschool_km', 'trc_sqm_5000', 'area_m',

                    # TODO check
                    # 'metro_min_avto',
                    # 'max_floor',
                    # 'is_prise_more',
                    # 'is_prise_less',
                    'is_na_floor',
                    'is_zero_floor',
                    'is_first_floor',

                    'office_sqm_5000',

                    # May be senseless
                    'industrial_km',

                    # Almost senseless
                    'park_km',
                    # 'office_sqm_5000',
                    'prom_part_5000',
                    # 'kremlin_km',

                    # Around 0
                    'public_transport_station_min_walk',
                    'school_km',
                    'state', 'ecology',
                    # Trash
                    'sale_month', 'sale_year',
                    'build_count_before_1920', 'build_count_1921-1945', 'build_count_1946-1970',
                    'build_count_1971-1995', 'build_count_after_1995',
                    # Does not improve model
                    'fixed_basket', 'metro_min_avto',
                    'ppi',
                    'kitch_sq',
                    'floor',
                    # 'product_type',
                    # Artificial
                    'pbm', 'pbm_usd',
                    'max_floor'
                    ], axis=1)


# print(train.head(3))
# sys.exit(0)

# an_xgb('_filtered_2', 1)
# sys.exit(0)


def cv(train):
    classifiers = [
        # xgb.XGBRegressor()
        # LinearRegression(),
        # GradientBoostingRegressor(max_depth=6, n_estimators=320),
        # GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=1000),
        # GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=1000),
        GradientBoostingRegressor(n_estimators=140,
                                  max_depth=3,
                                  random_state=1000),
        # ? Robust LR,
         # , learning_rate = 0.15
        # RandomForestRegressor(),
        # Lasso(),
        # LogisticRegression(),
        # CustomPredict(),
        # KNeighborsClassifier(3),
        # SVC(probability=True),
        # DecisionTreeClassifier(),
        # RandomForestClassifier(),
        # AdaBoostClassifier(),
        # GaussianNB(),
        # LinearDiscriminantAnalysis(),
        # QuadraticDiscriminantAnalysis(),
        # LogisticRegression()
    ]

    log_cols = ["Classifier", "Accuracy"]
    log = pd.DataFrame(columns=log_cols)

    num_of_test_splits = 10
    # sss = StratifiedShuffleSplit(n_splits=num_of_test_splits, test_size=0.1, random_state=0)

    # "          'n_estimators': 200,\n",
    # "          'max_depth': 5,\n",
    # "          'min_child_weight': 100,\n",
    # "          'subsample': .9,\n",
    # "          'gamma': 1,\n",
    # "          'objective': 'reg:linear',\n",
    # "          'colsample_bytree': .8,\n",
    # "\n",
    # "          'nthread':3,\n",
    # "          'silent':1,\n",
    # "          'seed':27\n",

    # "xgb_params = {\n",
    # "        'learning_rate': 0.05, 'max_depth': 4,'subsample': 0.9,\n",
    # "        'colsample_bytree': 0.9,'objective': 'binary:logistic',\n",
    # "        'silent': 1, 'n_estimators':100, 'gamma':1,\n",
    # "        'min_child_weight':4\n",
    # "        }   \n",
    # "clf = xgb.XGBClassifier(**xgb_params, seed = 10)     "

    # xgb_params = {
    #     'eta': 0.05,
    #     'max_depth': 5,
    #     'subsample': 1.0,
    #     'colsample_bytree': 0.7,
    #     'objective': 'reg:linear',
    #     'eval_metric': 'rmse',
    #     'silent': 1
    # }
    #
    # # Uncomment to tune XGB `num_boost_rounds`
    # partial_model = xgb.train(xgb_params, dtrain, num_boost_round=1000, evals=[(dval, 'val')],
    #                           early_stopping_rounds=20, verbose_eval=20)
    #
    # num_boost_round = partial_model.best_iteration


    source = train
    train = train.drop(['sub_area_origin'], axis=1)
    train = train.values
    # train = train[0:1000:, ]

    # Get all except of price_doc
    X = train[0::, :-1:]
    # Get only price_doc column
    y = train[0::, -1]

    # X = train.ix[:, train.columns == 'price_doc']
    # y = train.ix[:, train.columns != 'price_doc']

    # Choose classifier
    acc_dict = {}

    kf = KFold(n_splits=num_of_test_splits)
    kf.get_n_splits(X)

    i = 0
    print("Total len: ", len(train))
    r2 = []
    sle = []
    med = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # print("Set:", i, "; [", test_index[0], " .. ", test_index[-1], ']')
        print("Set:", i, ':')
        i += 1

        for clf in classifiers:
            name = clf.__class__.__name__
            clf.fit(X_train, y_train)
            y_predicted = clf.predict(X_test)
            mse = mean_squared_error(y_test, y_predicted)

            # print('Coefficients:', clf.coef_)
            print(
                'RMSLE: %.2f: ' % rmsle_vectorized(y_test, y_predicted),
                '; R2: %.2f' % clf.score(X_test, y_test), '; ',
                "mse: ~ [%.2f]" % (round(mse / 10 ** 10)))
            print()
            # , 'ma: %.2f' % median_absolute_error(y_test, y_predicted))

            r2.append(round(r2_score(y_test, y_predicted), 2))
            sle.append(rmsle_vectorized(y_test, y_predicted))
            med.append(round(median_absolute_error(y_test, y_predicted) / 10 ** 3))

            # scores_mse = cross_validation.cross_val_score(clf, x, y, scoring='mean_squared_error', cv=10)

            if i == 10:
                plt.plot(X_test, clf.predict(X_test), color='blue', linewidth=3)

                sample = source.iloc[test_index]
                sample['price_predicted'] = y_predicted
                sample['price_predicted'] = sample['price_predicted'].astype(int)
                sample['price_doc'] = sample['price_doc'].astype(int)
                sample['price_diff'] = np.round(np.abs(sample['price_doc']
                                              - sample['price_predicted']) / sample['price_doc'], 2)
                sample = sample.sort(['price_diff'], ascending=[0])
                sample = sample[[
                    'product_type', 'price_doc', 'price_predicted',
                    'full_sq', 'sub_area_origin', 'build_year',
                    'kremlin_km',
                    # 'ecology', 'prom_part_5000', 'office_sqm_5000', 'usdrub',
                    'bi_val', 'price_diff'
                ]]

                sample.to_csv("check_predictions.csv", index=False)

            # Plot outputs
            # if i == 1 or i == 10:
            #     plt.plot(X_test, clf.predict(X_test), color='blue', linewidth=3)
            #     plt.xticks(())
            #     plt.yticks(())
            #     plt.show()

            # if name in acc_dict:
            #     acc_dict[name] += acc
            # else:
            #     acc_dict[name] = acc
                # plt.scatter(X_test, y_test, color='black')

    print(pd.Series(clf.feature_importances_, index=predict_columns))
    # pd.concat([s1, s2], axis=1)

    # print(clf.feature_importances_)
    # print("Predict by columns: ", predict_columns)
    print("Avg: RMSLE:", sum(sle) / len(sle), " [", min(sle), "..", max(sle), ']',
          "Full list: ", sle)
    print("Avg: R2:", sum(r2) / len(r2), " [", min(r2), "..", max(r2), ']',
          "Full list: ", r2)
    print("Avg: med", sum(med) / len(med), " [", min(med), "..", max(med), ']',
          "Full list:", med)
    print()


def predict(data, train, predict_columns, is_write_result=None):
    # TODO ! Get validation for full_sq in wider range !!!
    test = format_data(data, True)
    train = train.drop(['sub_area_origin'], axis=1)
    train = train.values
    X = train[0::, :-1:]
    y = train[0::, -1]

    # regr = LinearRegression()
    # regr = GradientBoostingRegressor(max_depth=4, n_estimators=250, random_state=1000)
    # regr = GradientBoostingRegressor(max_depth=4, n_estimators=250)
    # regr = GradientBoostingRegressor(n_estimators=100, max_depth=8)
    regr = GradientBoostingRegressor(n_estimators=140,
                                  max_depth=3,
                                  random_state=1000)
    # regr = GradientBoostingRegressor(n_estimators=200, max_depth=5)
    # Lasso(),
    # LogisticRegression(),
    regr.fit(X, y)

    print(regr.feature_importances_)

    print("Predict by columns: ", predict_columns)

    # ! crossvsl
    # http://www.xavierdupre.fr/app/pymyinstall/helpsphinx/notebooks/example_xgboost.html

    # xgb_params = {
    #     'max_depth': 8,
    #     'subsample': 0.7,
    #     'colsample_bytree': 0.7,
    #     'objective': 'reg:linear',
    #     'silent': 0,
    #     'learning_rate': 0.05
    # }
    # **xgb_params

    # clf = xgb.XGBClassifier()
    # clf.fit(X, y)
    # predicted_price = clf.predict(test[predict_columns])

    predicted_price = regr.predict(test[predict_columns])

    test['price_doc'] = predicted_price
    # test['price_doc'] = predicted_price + 800000
    # test['price_doc_usd'] = predicted_price

    # print(test[test['kremlin_km'] == 14])
    # print(test[test['id'] == 30803])
    # print(test[test['price_doc'] < 0][['price_doc', 'kremlin_km']])

    if len(test[test['price_doc'] < 500000]):
        print(test[test['price_doc'] < 500000]
              [['price_doc',
                'usdrub', 'full_sq', 'kremlin_km', 'metro_min_avto',
                'sub_area', 'sub_area_origin',
                'build_year', 'state', 'ecology'
                ]])

    l = np.percentile(test['price_doc'].values, 0.5)
    r = np.percentile(test['price_doc'].values, 99.5)

    print(l, r)

    test.loc[test['price_doc'] < 600000, 'price_doc'] = 2550000
    plt.hist(test['price_doc'], bins='auto')
    plt.show()

    # print('Count with < 20 000 prices', len(test[test['price_doc_usd'] < 20000]))
    # print('Count with < 10 000 prices', len(test[test['price_doc_usd'] < 10000]))
    # print('Count with < 10 000 prices', len(test[test['price_doc_usd'] < 0]))
    # print('Avg predicted price ', np.mean(test['price_doc_usd']))
    #
    # mean = np.mean(test['price_doc_usd'])
    # test.loc[test['price_doc_usd'] < 0, 'price_doc_usd'] = mean
    # test.loc[test['price_doc_usd'] < 10000, 'price_doc_usd'] = mean / 3
    # *test['usdrub']

    # test['price_doc'] *= test['cpi']
    test['price_doc'] = round(test['price_doc'], 2)
    # test['price_doc'] = round(test['price_doc_usd'], 2)

    # print('Count with < 0 prices', len(test[test['price_doc_usd'] < 0]))
    if len(test[test['price_doc'] < 0]):
        print('Count with < 0 prices', len(test[test['price_doc'] < 0]))
    # print('Avg predicted price ', np.mean(test['price_doc_usd']))

    if is_write_result:
        submit = test[['id', 'price_doc']]
        submit.to_csv("submission7.csv", index=False)

    # submission = pd.DataFrame({'PassengerId': full_data[1][['PassengerId']],
    #                            'Survived': result_svc})
    # test = pd.merge(macro, test, how='right', on=['timestamp'])


def predict_my(train, predict_columns):
    # test = format_data(data)
    my_test = pd.read_csv(dir_path + '/my_test.csv', header=0)
    my_test.loc[np.isnan(my_test['min_walk_m']), 'min_walk_m'] = 3000
    my_test['ppi'] = 630
    my_test['usdrub'] = 57
    test = my_test

    train = train.values
    X = train[0::, :-1:]
    y = train[0::, -1]

    # Create linear regression object
    # regr = LinearRegression()
    regr = GradientBoostingRegressor()
    regr.fit(X, y)

    print("Predict by columns: ", predict_columns)

    # ! crossvsl
    # http://www.xavierdupre.fr/app/pymyinstall/helpsphinx/notebooks/example_xgboost.html

    # xgb_params = {
    #     'max_depth': 8,
    #     'subsample': 0.7,
    #     'colsample_bytree': 0.7,
    #     'objective': 'reg:linear',
    #     'silent': 0,
    #     'learning_rate': 0.05
    # }
    # **xgb_params

    # clf = xgb.XGBClassifier()
    # clf.fit(X, y)
    # predicted_price = clf.predict(test[predict_columns])

    predicted_price = regr.predict(test[predict_columns])

    # Convert to RUB
    test['price_doc_predict'] = predicted_price
    test['price_doc_predict'] = test['price_doc_predict'].astype(int)

    show_correlation(test, 'price_doc_predict', 'price_doc')

    # plt.hist(test['price_doc_predict'], bins='auto')
    # plt.show()

    print(test[['price_doc', 'price_doc_predict', 'num_rooms', 'full_sq',
                'min_walk_m', 'floor', 'max_floor']])


predict_columns = list(train)[0:-2]

print(predict_columns)
cv(train)

predict(test, train, predict_columns, 1)
# predict_my(train, predict_columns)
sys.exit()

# check_mean(train, 'num_room')

# print(stats.pearsonr(train[['full_sq']], train[['price_doc']]))

# print(train.head())
# print(train.groupby(['state'])['price_doc'].mean())

# 3. Regression check of multiple factors
# cw_lm = ols('price_doc ~ brent + full_sq + state + product_type'
#             , data=train).fit()
# ' + build_year '
# print(sm.stats.anova_lm(cw_lm, typ=2))
# sys.exit(0)

# g = sns.pairplot(train, diag_kind="kde")
# g = sns.pairplot(train[['build_year', 'price_doc']], kind="reg")
# plt.show()

# g = sns.pairplot(train, kind="reg")
# plt.savefig(os.path.basename(__file__) + '2.reg.png')
# sys.exit(0)

# g = sns.pairplot(train, kind="reg")
# plt.show()

# props = macro[['oil_urals', 'brent', 'salary']]
# props = props[~np.isnan(props['salary'])]
# props = props[~np.isnan(props['brent'])]

# print(stats.pearsonr(props[['oil_urals']], props[['brent']]))
# print(stats.pearsonr(props[['oil_urals']], props[['salary']]))

# plt.hist(props['salary'], bins='auto') plt.show()

# Visual check of data
# g = sns.pairplot(props, diag_kind="kde")
# plt.savefig(os.path.basename(__file__) + '.kde.png')
# g = sns.pairplot(data, kind="reg")
# plt.savefig(os.path.basename(__file__) + '.reg.png')

# 2. Manual check of factors
# print(data[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean())
# print(data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())

# print(macro.head())

# GradientBoostingMachine	0.65057
# RandomForest Gini	0.75107
# RandomForest Entropy	0.75222
# ExtraTrees Entropy	0.75524
# ExtraTrees Gini (Best)	0.75571
# Voting Ensemble (Democracy)	0.75337
# Voting Ensemble (3*Best vs. Rest)	0.75667
# ! MLWave

# data = pd.DataFrame({'expression_a': x1[1:], 'expression_b': y1[1:],
#                      'factor_a': ha[1:], 'factor_b': mo[1:]})

# TODO analyze different predictors
# for clf in acc_dict:
#     acc_dict[clf] /= num_of_test_splits
#     log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)
#     log = log.append(log_entry)

# log = log.sort(['Accuracy'], ascending=[0])
# print(log)

# Plot outputs
# plt.scatter(x_test, y_test, color='black')
# plt.plot(x_test, model.predict(x_test), color='blue', linewidth=3)

# plt.xticks(())
# plt.yticks(())
#
# plt.show()


# loo = cross_validation.KFold(len(y), n_folds=num_of_test_splits)
# regr = LinearRegression()
# scores = cross_validation.cross_val_score(regr, X, y,
#                                           scoring='mean_squared_error', cv=loo,)
# print('Coefficients: \n', regr.coef_)
# print("Residual sum of squares: %.2f"
#     % np.mean((regr.predict(test_x) - test_y) ** 2))
# #Explained variance score: 1 is perfect prediction
# print('Variance score: %.2f' % regr.score(test_x, test_y))


# plt.xlabel('Accuracy')
# plt.title('Classifier Accuracy')

# sns.set_color_codes("muted")
# sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")
#
# plt.savefig('tit_bar.png')

# Explained variance score: 1 is perfect prediction
# print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

# Plot outputs
# plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
# plt.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue',
#          linewidth=3)

# plt.xticks(())
# plt.yticks(())
#
# plt.show()

# result_svc = _predict(SVC(), test)
# result_boost = _predict(GradientBoostingClassifier(), test)
# result_forest = _predict(RandomForestClassifier(), test)
# result_gaussian = _predict(GaussianNB(), test)
# result_logistic = _predict(LogisticRegression(), test)
# result_linear = _predict(LinearDiscriminantAnalysis(), test)
# result_neighbors = _predict(KNeighborsClassifier(3), test)

# print(stats.pearsonr(result_svc, result_neighbors),
#       stats.pearsonr(result_neighbors, result_gaussian),
#       stats.pearsonr(result_svc, result_gaussian),
#       )

# Boosting classifier
# feature_importance = clf.feature_importances_
## make importances relative to max importance
# feature_importance = 100.0 * (feature_importance / feature_importance.max())
# sorted_idx = np.argsort(feature_importance)
# pos = np.arange(sorted_idx.shape[0]) + .5
# plt.subplot(1, 2, 2)
# plt.barh(pos, feature_importance[sorted_idx], align='center')
# plt.yticks(pos, train.feature_names[sorted_idx])
# plt.xlabel('Relative Importance')
# plt.title('Variable Importance')
# plt.show()

# Fin #################################
sys.exit(0)


def prepare_sq(data, sq_filed_name):
    # Set new feature sq_is_na before delete?
    data.loc[data[sq_filed_name] <= 15, sq_filed_name] = np.NAN
    data.loc[data[sq_filed_name] > 250, sq_filed_name] = np.NAN
    # TODO whether to set average set
    set_av(data, sq_filed_name)

    # data.loc[(data[sq_filed_name] > 5) & (data[sq_filed_name] <= 30), sq_filed_name] = 1
    # data.loc[(data[sq_filed_name] > 30) & (data[sq_filed_name] <= 40), sq_filed_name] = 2
    # data.loc[(data[sq_filed_name] > 40) & (data[sq_filed_name] <= 50), sq_filed_name] = 3
    # data.loc[(data[sq_filed_name] > 50) & (data[sq_filed_name] <= 80), sq_filed_name] = 4
    # data.loc[(data[sq_filed_name] > 80) & (data[sq_filed_name] <= 100), sq_filed_name] = 5
    # data.loc[(data[sq_filed_name] > 100) & (data[sq_filed_name] <= 200), sq_filed_name] = 6
    # data.loc[(data[sq_filed_name] > 200) & (data[sq_filed_name] <= 300), sq_filed_name] = 7
    # data.loc[data[sq_filed_name] > 300, sq_filed_name] = 8
    # # data.loc[(data[sq_filed_name] > 300) & (data[sq_filed_name] <= 500), sq_filed_name] = 8
    # data.loc[data[sq_filed_name] > 500, sq_filed_name] = 9
    data[sq_filed_name] = data[sq_filed_name].astype(int)

    # set_av(train, 'workplaces_km')
    # raw_data['workplaces_km'] = raw_data['workplaces_km'].astype(int)
    # raw_data.loc[raw_data['workplaces_km'] <= 1, 'workplaces_km'] = 0
    # raw_data.loc[(raw_data['workplaces_km'] > 1) & (raw_data['workplaces_km'] <= 5), 'workplaces_km'] = 1
    # raw_data.loc[(raw_data['workplaces_km'] > 5) & (raw_data['workplaces_km'] <= 15), 'workplaces_km'] = 2
    # raw_data.loc[(raw_data['workplaces_km'] > 15) & (raw_data['workplaces_km'] <= 30), 'workplaces_km'] = 3
    # raw_data.loc[(raw_data['workplaces_km'] > 30) & (raw_data['workplaces_km'] <= 50), 'workplaces_km'] = 4
    # raw_data.loc[raw_data['workplaces_km'] > 50, 'workplaces_km'] = 5

    # prepare_sq(raw_data, 'full_sq')


class CustomPredict:
    def __init__(self):
        self.p1 = SVC()
        self.p2 = KNeighborsClassifier(3)
        self.p3 = GaussianNB()

    def fit(self, _x_train, _y_train):
        self.p1.fit(_x_train, _y_train)
        self.p2.fit(_x_train, _y_train)
        self.p3.fit(_x_train, _y_train)

    def predict(self, data):
        res_svc = _predict(self.p1, data)
        res_neighbors = _predict(self.p2, data)
        res_gaussian = _predict(self.p3, data)
        res = res_svc + res_neighbors + res_gaussian
        # return res[np.where(res < 3, 0, 1)]
        return res[np.where(res < 2, 0, 1)]

        # print(res_logistic)
        # print(res_linear)
        # print(res_neighbors)
        # print(res)
        # print(result_logistic)


def old_format_data(raw_data):

    # print(len(train[train['sale_year'] < train['build_year']]))

    raw_data.loc[raw_data['material'] == 3, 'material'] = np.NAN
    default_validate(raw_data, 'material', 1, 6)

    default_validate(raw_data, 'build_year', 1800, 2050, custom_default=2017)
    raw_data['build_year'] /= 20
    raw_data['build_year'] = raw_data['build_year'].astype(int)

    # Todo mark if square is not specified
    percentile_validate(raw_data, 'full_sq', skip_left_outlines=1)
    percentile_validate(raw_data, 'metro_min_avto')
    # percentile_validate(raw_data, 'life_sq', skip_left_outlines=1)
    default_validate(raw_data, 'max_floor', 1, 200)
    default_validate(raw_data, 'floor', 0, 200)
    default_validate(raw_data, 'num_room', 1, 10)
    # default_validate(raw_data, 'metro_min_walk', 0, 1000)

    default_validate(raw_data, 'state', 0, 4, 0)
    filter_subarea(raw_data)
    filter_kremlin_km(raw_data)

    raw_data['product_type'] = raw_data['product_type']\
        .map({'Investment': 0, 'OwnerOccupier': 1})
    raw_data['product_type'] = raw_data['product_type'].astype(int)

    # percentile_validate(raw_data, 'kindergarten_km')
    # percentile_validate(raw_data, 'prom_part_5000')
    # percentile_validate(raw_data, 'office_sqm_5000', skip_left_outlines=1)
    # percentile_validate(raw_data, 'indust_part')
    # percentile_validate(raw_data, 'school_km')
    # percentile_validate(raw_data, 'park_km')
    # percentile_validate(raw_data, 'industrial_km')
    # percentile_validate(raw_data, 'cemetery_km')
    # percentile_validate(raw_data, 'water_km')
    return raw_data
