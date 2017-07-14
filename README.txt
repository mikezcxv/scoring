Summary:
Averaging of 4 models with params tuning: LogisticRegression,
GradientBoostingClassifier, GaussianNB, RandomForestClassifier.
AUC as metric.
CV on every single model in final solution ~ 70-72%,
 for mixed model ~ 80%.


Main steps:
1. Primary data processing:
    - Imputing of NAN values with means (the impute strategy will be corrected later)
        for fn in {'Income', 'Assets', 'ResidenceType', 'EmploymentType', 'Debt'}:
            raw.loc[np.isnan(raw[fn]), fn] = np.mean(raw[fn])

    - Removal of noise property PhoneNumber (its the same for > 99% of records)
        - There is also some duplicate for one phone number. But it doesn't us any useful info.
        - We will try to extract phone code which may be connected with another feature.
    - Removal of 'ApplicationId'
        - Probably this var is connected with time period, which affects target var

3. Choosing of CV and metrics:
    - The classes of target var are unbalanced (good / bad = 2546 / 1017), so we will use 'auc'
    - CV: StratifiedShuffleSplit for 8 groups.

4. Approximate comparison of 10 classifiers on default parameters.
    KNeighborsClassifier(),
    SVC(random_state=42),
    DecisionTreeClassifier(random_state=42),
    RandomForestClassifier(random_state=42),
    AdaBoostClassifier(random_state=42),
    GradientBoostingClassifier(random_state=42),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression(random_state=42)

5. Data processing:
    - An attempt to group some data
    - Processing NAN Values
        print(raw[raw['Assets'] == 0][['Status', 'Assets']].groupby(['Status'], as_index=False).agg(['count', 'mean', 'min', 'max']))
        print(raw[raw['Assets'] > 0][['Status', 'Assets']].groupby(['Status'], as_index=False).agg(['count', 'mean', 'min', 'max']))
        print(raw[pd.isnull(raw['Assets'])].groupby(['Status'], as_index=False).agg(['count', 'mean', 'min', 'max']))

    Group with 0 Assets has significantly lower bad: 522 / 522 + 758 ~ 40.8%.
    In the second group: 479 / 479 + 1766
    0         522
    1         758
    Assets
            count         mean    min       max
    0         479  7841.102296  750.0  100000.0
    1        1766  8447.604190   18.0  300000.0

    There is bad/(bad+good) ~ 40.4% ratio in NAN group

    - Triing new feature 'AmountToPrice' : (raw['Price'] - raw['Amount']) / raw['Price']
                 count    mean   min    max
        0        1016  0.201676  0.0  0.900000
        1        2546  0.301620  0.0  0.932976
        Ks_2sampResult(statistic=0.22869747821192415, pvalue=1.0128290814307454e-33)
        Ttest_indResult(statistic=13.606874660779752, pvalue=3.8304446367378185e-41)

        'GradientBoostingClassifier': 0.70980392156862748 ->
        'GradientBoostingClassifier': 0.7198529411764707

        'RandomForestClassifier': 0.70024509803921575, ->
        'RandomForestClassifier': 0.71617647058823541

    - Exploring empty values of Income
                count        mean   min    max
        0         845  120.362130  16.0  959.0
        1        2420  149.222727   6.0  905.0
        Ks_2sampResult(statistic=0.16736226657842163, pvalue=2.9508252826361461e-18)
        Ttest_indResult(statistic=8.9145496964721804, pvalue=7.9800098735681236e-19)

        - Group with NAN income % bad 171
                     count
        0            171
        1            126
            % bad    57
        - Adding property RelAmountToPrice
                count      mean   min    max
        0        1016  1.904361  0.0  13.276353
        1        2546  2.710033  0.0  19.943020
        Ks_2sampResult(statistic=0.25396793488009606, pvalue=1.7413436105564489e-41)
        Ttest_indResult(statistic=13.015588173281058, pvalue=7.2028390487614717e-38)

        - There is no data in test set without EmploymentType.
         Lets replace those data in train set to 2 (group with the largest %bad)

        - 'Debt' > 0. Skip this var
        Ks_2sampResult(statistic=0.084658917100030595, pvalue=0.32501539608842622)
        Ttest_indResult(statistic=-1.4541758247832501, pvalue=0.14639320093987193)

6. Simplification of model: dropped some features with low prediction power
7. Data handling:
    - Merge some small (< 30) ranges in Time feature
    - Merge small (< 20) type 4 of ResidenceType with sibling


8. Averaging of 4-th good and slightly different classifiers
9. Params tuning.