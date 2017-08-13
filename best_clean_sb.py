import re
# import numpy as np
from helpers.my_help import *
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor


def filter_full_sq(data, is_production=None):
    # mean_life_to_full_square = np.mean(raw_data['life_sq']) / np.mean(data['full_sq'])

    right_outline = 99.998 if is_production else 99.9
    rlimit = np.percentile(data['full_sq'].values, right_outline)
    llimit = 5
    # -0.005
    # raw_data['is_large_full_sq'] = (raw_data['full_sq'] > rlimit)
    # 0.003
    # raw_data['is_less_full_sq'] = (raw_data['full_sq'] < llimit)

    # data.insert(0, 'is_large_full_sq', np.NAN)
    # data['is_large_full_sq'] = (data['full_sq'] > rlimit)

    data.loc[data['full_sq'] <= llimit, 'full_sq'] = np.NAN
    data.loc[data['full_sq'] > rlimit, 'full_sq'] = rlimit

    return

    # TODO revert for not xgb
    data.loc[np.isnan(data['full_sq']), 'full_sq'] = np.mean(data['full_sq'])

    # print(raw_data[raw_data['full_sq'] < raw_data['life_sq']]
    #       [['sale_year', 'full_sq', 'life_sq', 'floor', 'max_floor', 'price_doc']])

    # raw_data['full_sq_log'] = np.log10(raw_data['full_sq'])
    # raw_data['full_sq_log_e'] = np.log(raw_data['full_sq'])
    # raw_data['full_sq'] = raw_data['full_sq'] ** 1.45
    # describe(raw_data, 'full_sq', nlo=30)
    # show_bar_plot(raw_data, 'full_sq', 'price_doc')
    # get_correlation(raw_data, 'full_sq', 'price_doc')
    # get_correlation(raw_data, 'full_sq_log_2', 'price_doc')
    # get_correlation(raw_data, 'full_sq_log', 'price_doc')
    # get_correlation(raw_data, 'full_sq_log_e', 'price_doc')

    # print(len(train[train['full_sq'] < llimit]))
    # print(len(train[train['full_sq'] > rlimit]))
    # print(len(train[np.isnan(train['full_sq'])]))

    # print(llimit, rlimit)

    # percentile_validate(raw_data, 'full_sq')


def create_bi_val(raw_data, offset=''):
    suff = '_macro' + str(offset) if offset else ''
    raw_data.insert(0, 'bi_val' + str(offset), (.55 * raw_data['usdrub' + suff]
                                           + .45 * raw_data['eurrub' + suff]))

    # show_correlations(raw_data, 'usdrub')
    # show_correlations(raw_data, 'eurrub')
    # show_correlations(raw_data, 'bi_val')
    # show_feature_over_time(raw_data, 'usdrub')
    # show_feature_over_time(raw_data, 'bi_val')
    # show_pair_plot(raw_data, ['bi_val', 'eurrub', 'usdrub', 'price_doc'])


def set_year(data):
    data['sale_year'] = data['timestamp'].apply(lambda x: int(re.compile('\d+').match(x).group()))
    data['sale_month'] = data['timestamp'].apply(lambda x: int(re.compile('\d+-(\d+)').match(x).group(1)))
    # data['sale_year_month'] = data['timestamp']\
    #     .apply(lambda x: re.compile('(\d+-\d+)').match(x).group(1))


def filter_subarea(raw_data):
    raw_data.loc[:, "sub_area"] = raw_data.sub_area.\
        str.replace(" ", "").str.replace("\'", "").str.replace("-", "")

    raw_data['sub_area_origin'] = raw_data['sub_area']

    # show_bar_plot(raw_data, 'sale_year_month', 'price_doc')
    # print(raw_data['sub_area'])

    # raw_data = raw_data.drop(['pbm_usd'], axis=1)
    # print(raw_data[raw_data['pbm_usd'] == 13802])

    # k = raw_data[(raw_data['sub_area'] == 'Arbat')
    # & (raw_data['ppm'] > 150000)
    # print(k[['full_sq', 'num_room', 'price_doc', 'ppm', 'year_old']])
    # show_hist(k['price_doc'])
    # asd()
    # print(raw_data[(raw_data['sub_area'] == 'Donskoe')
    #                & (raw_data['ppm'] > 150000)]
    #       [['full_sq', 'num_room', 'price_doc', 'ppm']])
    # asd()

    # print(stat[stat['std'] > 80432])
    # normalized = aggr['ppm'] - means
        # np.std(

        # / np.linalg.norm(aggr['price_doc']))

    # ['ppm', 'sub_area']\
    #     .agg(lambda x: (x - np.mean(x)) / np.std(x))
        # .agg(['std', 'max', 'count'])

    # di = {}
    # for k, gp in aggr['ppm']['max'].iteritems():
    #     di[k] = int(gp)

    # Get sigmas
    # raw_data['ppm'] = raw_data['price_doc'] / (raw_data['full_sq'])
    # aggr = raw_data.groupby(['sub_area'])
    #
    # stat = aggr['ppm'].agg(['std', 'min', 'mean', 'max', 'count'])
    # stat['sigma_left'] = stat['mean'] - stat['std']
    # stat['sigma_right'] = stat['mean'] + stat['std']
    # stat['sigma_left2'] = stat['mean'] - 2 * stat['std']
    # stat['sigma_right2'] = stat['mean'] + 2 * stat['std']
    #
    # for h in ['left', 'right', 'left2', 'right2']:
    #     v = 'sigma_' + h
    #     m = dict(zip(stat.index.values, stat[v]))
    #     print(v, '= ', m, sep='')
    #
    # asd()

    min_price = {'Troickijokrug': 12345, 'Hovrino': 21276, 'Beskudnikovskoe': 15873, 'Lomonosovskoe': 16229, 'Izmajlovo': 16129, 'Presnenskoe': 14492, 'Vostochnoe': 82222, 'MarinaRoshha': 12987, 'Novokosino': 13513, 'OrehovoBorisovoSevernoe': 13378, 'VostochnoeIzmajlovo': 20833, 'Zamoskvoreche': 20408, 'Preobrazhenskoe': 14285, 'Nagornoe': 19607, 'Danilovskoe': 17241, 'Zjuzino': 12060, 'FilevskijPark': 17068, 'Mitino': 13888, 'Bogorodskoe': 14705, 'Jakimanka': 12500, 'PoseleniePervomajskoe': 21532, 'Kapotnja': 17460, 'Meshhanskoe': 15147, 'Akademicheskoe': 14558, 'Severnoe': 18518, 'Dmitrovskoe': 16129, 'Rostokino': 14285, 'PoselenieVnukovskoe': 44880, 'Ivanovskoe': 16949, 'PoselenieVoskresenskoe': 26913, 'Losinoostrovskoe': 13513, 'Nekrasovka': 12099, 'Konkovo': 19411, 'Cheremushki': 18867, 'OrehovoBorisovoJuzhnoe': 16666, 'Dorogomilovo': 17543, 'PoselenieKlenovskoe': 23255, 'Brateevo': 13888, 'HoroshevoMnevniki': 12345, 'Kurkino': 25641, 'ChertanovoJuzhnoe': 13157, 'Novogireevo': 17857, 'ProspektVernadskogo': 22000, 'VostochnoeDegunino': 13333, 'Sokol': 16129, 'TroparevoNikulino': 12820, 'PoselenieRjazanovskoe': 22000, 'Pechatniki': 19411, 'Arbat': 91666, 'BirjulevoVostochnoe': 15151, 'Krylatskoe': 26315, 'Tekstilshhiki': 13513, 'Savelovskoe': 17543, 'OchakovoMatveevskoe': 13414, 'Caricyno': 19607, 'Solncevo': 13888, 'Goljanovo': 12658, 'Jasenevo': 14925, 'SevernoeTushino': 16666, 'PoselenieShherbinka': 21505, 'Marino': 12500, 'Veshnjaki': 16750, 'Basmannoe': 15000, 'Marfino': 12987, 'Ajeroport': 25000, 'NagatinskijZaton': 15151, 'Rjazanskij': 15384, 'Savelki': 13750, 'Koptevo': 13888, 'Metrogorodok': 27027, 'Sokolniki': 36363, 'PoselenieKievskij': 60714, 'ChertanovoSevernoe': 22727, 'Lianozovo': 15873, 'Jaroslavskoe': 16650, 'NovoPeredelkino': 13333, 'Nizhegorodskoe': 17857, 'Vnukovo': 18333, 'FiliDavydkovo': 17155, 'PoselenieKokoshkino': 15625, 'PoselenieSosenskoe': 20833, 'Otradnoe': 12692, 'Strogino': 15625, 'PoselenieMoskovskij': 40000, 'Alekseevskoe': 13333, 'Butyrskoe': 13333, 'Tverskoe': 27777, 'PokrovskoeStreshnevo': 29411, 'PoselenieMihajlovoJarcevskoe': 72500, 'JuzhnoeMedvedkovo': 13200, 'BirjulevoZapadnoe': 14705, 'Sviblovo': 17543, 'Kuzminki': 14285, 'Molzhaninovskoe': 16000, 'PoselenieFilimonkovskoe': 23571, 'Gagarinskoe': 28571, 'SevernoeIzmajlovo': 17678, 'JuzhnoeButovo': 13513, 'SevernoeMedvedkovo': 13157, 'PoselenieVoronovskoe': 42803, 'Bibirevo': 16393, 'Silino': 19230, 'PoselenieRogovskoe': 39925, 'Golovinskoe': 13333, 'SokolinajaGora': 12226, 'TeplyjStan': 14271, 'JuzhnoeTushino': 16129, 'Matushkino': 12820, 'Altufevskoe': 12500, 'KosinoUhtomskoe': 12531, 'Babushkinskoe': 18867, 'VyhinoZhulebino': 16779, 'Timirjazevskoe': 17857, 'Taganskoe': 13157, 'Hamovniki': 42553, 'PoselenieDesjonovskoe': 17678, 'Krjukovo': 12487, 'NagatinoSadovniki': 18867, 'Zjablikovo': 16666, 'Donskoe': 12820, 'Perovo': 12987, 'Juzhnoportovoe': 22200, 'PoselenieMarushkinskoe': 21739, 'Ramenki': 13513, 'Kuncevo': 16438, 'Lefortovo': 21818, 'Krasnoselskoe': 28571, 'Vojkovskoe': 15625, 'Ljublino': 12048, 'Mozhajskoe': 20408, 'ZapadnoeDegunino': 13513, 'StaroeKrjukovo': 13157, 'PoselenieMosrentgen': 30937, 'Begovoe': 15151, 'ChertanovoCentralnoe': 15468, 'PoselenieKrasnopahorskoe': 16393, 'PoselenieShhapovskoe': 43196, 'Obruchevskoe': 16666, 'Shhukino': 13200, 'ErrorRaionPoselenie': 29478, 'Levoberezhnoe': 12658, 'Ostankinskoe': 17543, 'Kotlovka': 17241, 'Horoshevskoe': 14866, 'SevernoeButovo': 12820, 'MoskvorecheSaburovo': 12345, 'PoselenieNovofedorovskoe': 39796}
    mean_price = {'PoselenieKievskij': 127731, 'PoseleniePervomajskoe': 63918, 'Danilovskoe': 174391, 'OrehovoBorisovoSevernoe': 132978, 'Sviblovo': 161945, 'Levoberezhnoe': 147598, 'Caricyno': 145293, 'PoselenieDesjonovskoe': 91998, 'Strogino': 168834, 'Marfino': 141521, 'Losinoostrovskoe': 146407, 'ProspektVernadskogo': 191214, 'Timirjazevskoe': 151137, 'PoselenieMarushkinskoe': 70473, 'Preobrazhenskoe': 151910, 'Jakimanka': 177261, 'Bibirevo': 137143, 'Silino': 106077, 'Savelovskoe': 168447, 'VyhinoZhulebino': 134509, 'Gagarinskoe': 206092, 'Jasenevo': 150189, 'PoselenieVnukovskoe': 111043, 'Vojkovskoe': 156298, 'JuzhnoeMedvedkovo': 131759, 'NovoPeredelkino': 109627, 'Donskoe': 191171, 'Juzhnoportovoe': 164670, 'Basmannoe': 193732, 'Novogireevo': 144940, 'Tverskoe': 241119, 'Otradnoe': 144044, 'KosinoUhtomskoe': 123312, 'PoselenieVoskresenskoe': 95301, 'NagatinskijZaton': 168377, 'Lefortovo': 154613, 'PoselenieSosenskoe': 94496, 'Goljanovo': 129329, 'Obruchevskoe': 162946, 'Nagornoe': 177951, 'Krylatskoe': 196099, 'PoselenieRogovskoe': 52241, 'Taganskoe': 179282, 'PoselenieKrasnopahorskoe': 62010, 'Presnenskoe': 203466, 'Marino': 141365, 'Matushkino': 103663, 'PoselenieVoronovskoe': 73060, 'Jaroslavskoe': 129632, 'OchakovoMatveevskoe': 146012, 'PokrovskoeStreshnevo': 161109, 'FiliDavydkovo': 162304, 'VostochnoeDegunino': 136241, 'ChertanovoJuzhnoe': 144005, 'Rjazanskij': 139703, 'Mitino': 148689, 'Zamoskvoreche': 236259, 'Metrogorodok': 137114, 'JuzhnoeButovo': 127306, 'PoselenieKokoshkino': 71441, 'OrehovoBorisovoJuzhnoe': 141188, 'Mozhajskoe': 143783, 'SevernoeIzmajlovo': 139362, 'Begovoe': 194571, 'Koptevo': 149507, 'ChertanovoCentralnoe': 146894, 'MoskvorecheSaburovo': 150705, 'HoroshevoMnevniki': 157191, 'Tekstilshhiki': 136615, 'Rostokino': 153140, 'Alekseevskoe': 177749, 'Ajeroport': 178072, 'Veshnjaki': 128616, 'Kuncevo': 156204, 'Kuzminki': 139478, 'PoselenieShhapovskoe': 45035, 'Nekrasovka': 100088, 'BirjulevoVostochnoe': 118224, 'Hamovniki': 277649, 'PoselenieMosrentgen': 99360, 'Shhukino': 171975, 'Ivanovskoe': 127227, 'PoselenieFilimonkovskoe': 74344, 'PoselenieKlenovskoe': 23255, 'Krjukovo': 105430, 'FilevskijPark': 169265, 'SevernoeButovo': 146767, 'Babushkinskoe': 152417, 'PoselenieMihajlovoJarcevskoe': 72500, 'SevernoeMedvedkovo': 145109, 'ErrorRaionPoselenie': 111258, 'Lomonosovskoe': 201075, 'Akademicheskoe': 176540, 'Dmitrovskoe': 131086, 'Ostankinskoe': 178635, 'Brateevo': 133901, 'Ramenki': 184088, 'Perovo': 136971, 'TeplyjStan': 149872, 'StaroeKrjukovo': 98523, 'Horoshevskoe': 187425, 'Bogorodskoe': 140742, 'Sokolniki': 199902, 'NagatinoSadovniki': 156314, 'Izmajlovo': 148914, 'PoselenieMoskovskij': 95413, 'Cheremushki': 180006, 'Troickijokrug': 82940, 'Savelki': 105914, 'Vnukovo': 107753, 'Hovrino': 145832, 'Meshhanskoe': 203911, 'Zjablikovo': 141254, 'Lianozovo': 135715, 'Nizhegorodskoe': 133649, 'Severnoe': 110018, 'SokolinajaGora': 149693, 'BirjulevoZapadnoe': 118093, 'MarinaRoshha': 163610, 'Solncevo': 114383, 'PoselenieNovofedorovskoe': 56341, 'Butyrskoe': 150873, 'Vostochnoe': 121639, 'Kotlovka': 159919, 'Dorogomilovo': 224185, 'Kapotnja': 113681, 'TroparevoNikulino': 168554, 'VostochnoeIzmajlovo': 138495, 'ChertanovoSevernoe': 152669, 'Pechatniki': 131957, 'Golovinskoe': 139659, 'PoselenieRjazanovskoe': 85022, 'Ljublino': 136744, 'Zjuzino': 157247, 'Konkovo': 170781, 'Krasnoselskoe': 194963, 'Altufevskoe': 122475, 'JuzhnoeTushino': 143364, 'ZapadnoeDegunino': 101933, 'Beskudnikovskoe': 135254, 'Sokol': 164045, 'Arbat': 270062, 'Molzhaninovskoe': 39543, 'PoselenieShherbinka': 85418, 'Novokosino': 137061, 'Kurkino': 146546, 'SevernoeTushino': 157306}
    max_price = {'PoselenieMoskovskij': 210798, 'PoselenieFilimonkovskoe': 137790, 'Shhukino': 340677, 'Altufevskoe': 229411, 'Alekseevskoe': 320787, 'Sokol': 292272, 'Novokosino': 182191, 'Preobrazhenskoe': 225806, 'Sokolniki': 424672, 'Kapotnja': 168000, 'PoselenieNovofedorovskoe': 103448, 'ChertanovoJuzhnoe': 209103, 'Hovrino': 225925, 'Pechatniki': 193750, 'Veshnjaki': 198039, 'Dorogomilovo': 435483, 'SevernoeIzmajlovo': 223809, 'Juzhnoportovoe': 290740, 'ProspektVernadskogo': 539393, 'Horoshevskoe': 334408, 'OrehovoBorisovoJuzhnoe': 200000, 'Krasnoselskoe': 312500, 'PoselenieRogovskoe': 176315, 'Nizhegorodskoe': 208333, 'Ivanovskoe': 200000, 'FiliDavydkovo': 326086, 'NagatinskijZaton': 233823, 'Matushkino': 160000, 'PoselenieKrasnopahorskoe': 113936, 'Marfino': 243518, 'Zamoskvoreche': 335843, 'Nagornoe': 261907, 'Izmajlovo': 279691, 'Tekstilshhiki': 325581, 'Otradnoe': 230519, 'Timirjazevskoe': 268867, 'Metrogorodok': 194444, 'ErrorRaionPoselenie': 367686, 'PokrovskoeStreshnevo': 382535, 'Strogino': 364431, 'Basmannoe': 391304, 'PoselenieMosrentgen': 148571, 'JuzhnoeTushino': 269230, 'Begovoe': 312195, 'Brateevo': 196969, 'SevernoeMedvedkovo': 219047, 'PoselenieMarushkinskoe': 109090, 'SevernoeButovo': 223684, 'PoselenieShhapovskoe': 46875, 'Danilovskoe': 432285, 'Lianozovo': 206944, 'TeplyjStan': 237735, 'Ostankinskoe': 313750, 'Perovo': 290000, 'Beskudnikovskoe': 203703, 'Krjukovo': 155555, 'Levoberezhnoe': 222058, 'Kotlovka': 254237, 'Ajeroport': 471851, 'Savelovskoe': 263566, 'Rjazanskij': 214285, 'VostochnoeIzmajlovo': 292307, 'Molzhaninovskoe': 83333, 'Caricyno': 264705, 'Ramenki': 492249, 'MoskvorecheSaburovo': 243589, 'Presnenskoe': 432374, 'Gagarinskoe': 349206, 'Rostokino': 272463, 'Obruchevskoe': 389221, 'Troickijokrug': 141666, 'Hamovniki': 541666, 'Butyrskoe': 273437, 'Mitino': 247563, 'Zjuzino': 285714, 'BirjulevoVostochnoe': 205000, 'BirjulevoZapadnoe': 160294, 'TroparevoNikulino': 271428, 'SokolinajaGora': 230000, 'Lefortovo': 279166, 'Vnukovo': 153846, 'Babushkinskoe': 343155, 'Jaroslavskoe': 207662, 'Sviblovo': 236363, 'Lomonosovskoe': 323394, 'Jasenevo': 228947, 'PoselenieKlenovskoe': 23255, 'JuzhnoeButovo': 193750, 'Losinoostrovskoe': 202631, 'Zjablikovo': 192105, 'JuzhnoeMedvedkovo': 210256, 'ChertanovoCentralnoe': 459864, 'Silino': 179166, 'Solncevo': 207894, 'Nekrasovka': 232894, 'StaroeKrjukovo': 165517, 'ChertanovoSevernoe': 227631, 'Kuzminki': 255319, 'SevernoeTushino': 222017, 'Bibirevo': 191891, 'PoselenieShherbinka': 151724, 'Jakimanka': 379032, 'Koptevo': 297297, 'OchakovoMatveevskoe': 232456, 'PoselenieVoskresenskoe': 151390, 'Arbat': 451612, 'Severnoe': 167088, 'FilevskijPark': 274000, 'Taganskoe': 425531, 'Bogorodskoe': 314102, 'Konkovo': 268595, 'PoselenieRjazanovskoe': 138597, 'HoroshevoMnevniki': 320000, 'PoselenieDesjonovskoe': 125491, 'ZapadnoeDegunino': 180000, 'Vostochnoe': 150000, 'NovoPeredelkino': 164705, 'MarinaRoshha': 490445, 'VostochnoeDegunino': 208333, 'PoseleniePervomajskoe': 105882, 'Novogireevo': 206140, 'Meshhanskoe': 382258, 'PoselenieVnukovskoe': 170454, 'Mozhajskoe': 274509, 'Ljublino': 288888, 'PoselenieMihajlovoJarcevskoe': 72500, 'PoselenieVoronovskoe': 96725, 'Vojkovskoe': 234328, 'Golovinskoe': 216666, 'Donskoe': 532545, 'Cheremushki': 304878, 'PoselenieKokoshkino': 122580, 'Dmitrovskoe': 185526, 'NagatinoSadovniki': 256756, 'Krylatskoe': 321739, 'Goljanovo': 300000, 'PoselenieKievskij': 194749, 'OrehovoBorisovoSevernoe': 212121, 'Kuncevo': 368595, 'Kurkino': 230000, 'Savelki': 168750, 'KosinoUhtomskoe': 173684, 'Marino': 213157, 'Akademicheskoe': 300000, 'VyhinoZhulebino': 196710, 'PoselenieSosenskoe': 399826, 'Tverskoe': 443806}
    count = {'Kotlovka': 147, 'SokolinajaGora': 188, 'Novogireevo': 200, 'Solncevo': 419, 'Sviblovo': 131, 'NagatinoSadovniki': 158, 'Kurkino': 61, 'Rjazanskij': 181, 'Marfino': 85, 'PoselenieFilimonkovskoe': 551, 'NagatinskijZaton': 325, 'Juzhnoportovoe': 126, 'NovoPeredelkino': 201, 'Marino': 505, 'PoselenieKlenovskoe': 1, 'StaroeKrjukovo': 91, 'Basmannoe': 99, 'ErrorRaionPoselenie': 52, 'Dorogomilovo': 56, 'Brateevo': 182, 'Savelovskoe': 85, 'JuzhnoeTushino': 175, 'PoselenieMoskovskij': 966, 'OrehovoBorisovoJuzhnoe': 208, 'Altufevskoe': 68, 'Hamovniki': 90, 'Otradnoe': 358, 'Bogorodskoe': 304, 'Ljublino': 296, 'Krasnoselskoe': 37, 'Taganskoe': 173, 'Gagarinskoe': 78, 'Preobrazhenskoe': 151, 'Konkovo': 219, 'Kapotnja': 49, 'Krjukovo': 522, 'JuzhnoeButovo': 451, 'Ajeroport': 123, 'Mitino': 677, 'Matushkino': 111, 'Shhukino': 155, 'Sokolniki': 60, 'Zjuzino': 260, 'MarinaRoshha': 116, 'Molzhaninovskoe': 3, 'Arbat': 15, 'Tverskoe': 75, 'Ostankinskoe': 79, 'Lianozovo': 125, 'VostochnoeIzmajlovo': 153, 'Koptevo': 206, 'Nagornoe': 302, 'ChertanovoCentralnoe': 196, 'Vostochnoe': 7, 'Kuncevo': 186, 'Severnoe': 37, 'PoselenieKievskij': 2, 'MoskvorecheSaburovo': 99, 'Pechatniki': 192, 'Kuzminki': 221, 'Levoberezhnoe': 134, 'Beskudnikovskoe': 166, 'Izmajlovo': 300, 'Alekseevskoe': 99, 'Krylatskoe': 102, 'ChertanovoJuzhnoe': 272, 'HoroshevoMnevniki': 260, 'Danilovskoe': 199, 'Tekstilshhiki': 298, 'SevernoeMedvedkovo': 167, 'SevernoeIzmajlovo': 163, 'Ramenki': 241, 'Goljanovo': 293, 'Savelki': 102, 'PoselenieKokoshkino': 19, 'Zjablikovo': 127, 'Horoshevskoe': 134, 'FilevskijPark': 146, 'PoselenieShherbinka': 474, 'SevernoeTushino': 281, 'Losinoostrovskoe': 177, 'PoselenieMosrentgen': 19, 'Nekrasovka': 1718, 'PoselenieMarushkinskoe': 6, 'TeplyjStan': 164, 'FiliDavydkovo': 136, 'Metrogorodok': 58, 'Vojkovskoe': 131, 'PoselenieDesjonovskoe': 400, 'PoselenieNovofedorovskoe': 156, 'Timirjazevskoe': 154, 'Butyrskoe': 101, 'Veshnjaki': 212, 'PoselenieVnukovskoe': 1412, 'Meshhanskoe': 94, 'VyhinoZhulebino': 262, 'Babushkinskoe': 123, 'ChertanovoSevernoe': 228, 'PoselenieMihajlovoJarcevskoe': 1, 'Begovoe': 60, 'Novokosino': 138, 'SevernoeButovo': 181, 'Jaroslavskoe': 121, 'Cheremushki': 158, 'Dmitrovskoe': 174, 'Sokol': 74, 'PoselenieRogovskoe': 34, 'Strogino': 301, 'Presnenskoe': 189, 'BirjulevoVostochnoe': 267, 'Nizhegorodskoe': 77, 'PokrovskoeStreshnevo': 162, 'Mozhajskoe': 197, 'Jakimanka': 81, 'PoselenieSosenskoe': 1826, 'Rostokino': 66, 'Jasenevo': 237, 'BirjulevoZapadnoe': 115, 'Troickijokrug': 159, 'PoselenieKrasnopahorskoe': 30, 'Vnukovo': 43, 'Zamoskvoreche': 50, 'Hovrino': 178, 'Golovinskoe': 224, 'OrehovoBorisovoSevernoe': 206, 'PoselenieShhapovskoe': 2, 'Bibirevo': 230, 'ProspektVernadskogo': 100, 'OchakovoMatveevskoe': 255, 'PoseleniePervomajskoe': 142, 'Caricyno': 219, 'PoselenieVoskresenskoe': 738, 'PoselenieRjazanovskoe': 34, 'Perovo': 246, 'Donskoe': 135, 'JuzhnoeMedvedkovo': 143, 'KosinoUhtomskoe': 237, 'PoselenieVoronovskoe': 7, 'ZapadnoeDegunino': 409, 'Akademicheskoe': 214, 'Lomonosovskoe': 147, 'Lefortovo': 119, 'Silino': 100, 'Obruchevskoe': 182, 'TroparevoNikulino': 125, 'VostochnoeDegunino': 118, 'Ivanovskoe': 196}
    mean_price_usd = {'Begovoe': 5296, 'Marfino': 3929, 'PoselenieVoskresenskoe': 2852, 'Vostochnoe': 3512, 'Ljublino': 3788, 'Severnoe': 3063, 'Juzhnoportovoe': 4670, 'PokrovskoeStreshnevo': 4546, 'Jaroslavskoe': 3521, 'MoskvorecheSaburovo': 4170, 'Molzhaninovskoe': 763, 'Kapotnja': 3342, 'FilevskijPark': 4214, 'Vnukovo': 3139, 'JuzhnoeButovo': 3468, 'NovoPeredelkino': 3179, 'Rostokino': 4115, 'JuzhnoeMedvedkovo': 3664, 'Matushkino': 2868, 'PoselenieVoronovskoe': 2024, 'SevernoeMedvedkovo': 3895, 'Golovinskoe': 3821, 'ErrorRaionPoselenie': 2848, 'Basmannoe': 5369, 'PoselenieMihajlovoJarcevskoe': 2098, 'ProspektVernadskogo': 5350, 'Savelki': 2957, 'Nekrasovka': 2857, 'ChertanovoSevernoe': 4178, 'FiliDavydkovo': 4403, 'Ostankinskoe': 4855, 'Pechatniki': 3697, 'VostochnoeDegunino': 3748, 'Alekseevskoe': 4751, 'TroparevoNikulino': 4560, 'Horoshevskoe': 4905, 'Levoberezhnoe': 3752, 'PoselenieKievskij': 2651, 'PoselenieNovofedorovskoe': 1238, 'Caricyno': 4029, 'PoselenieShhapovskoe': 1341, 'Dorogomilovo': 6129, 'Strogino': 4781, 'Novokosino': 3708, 'Arbat': 7770, 'Donskoe': 5264, 'PoselenieKlenovskoe': 717, 'VostochnoeIzmajlovo': 3734, 'Dmitrovskoe': 3636, 'Hamovniki': 7796, 'PoselenieKrasnopahorskoe': 1785, 'Vojkovskoe': 4343, 'PoselenieVnukovskoe': 2737, 'Izmajlovo': 4139, 'Ajeroport': 4903, 'PoseleniePervomajskoe': 1681, 'Obruchevskoe': 4702, 'Timirjazevskoe': 4214, 'Sokolniki': 5700, 'Koptevo': 4167, 'Nizhegorodskoe': 3753, 'PoselenieRjazanovskoe': 2161, 'Krasnoselskoe': 5726, 'OrehovoBorisovoJuzhnoe': 3949, 'Lomonosovskoe': 5427, 'Babushkinskoe': 4340, 'BirjulevoZapadnoe': 3327, 'Veshnjaki': 3531, 'OrehovoBorisovoSevernoe': 3612, 'Zamoskvoreche': 6480, 'Losinoostrovskoe': 3963, 'SokolinajaGora': 4131, 'ChertanovoJuzhnoe': 4020, 'Danilovskoe': 4558, 'PoselenieSosenskoe': 2591, 'Altufevskoe': 3532, 'Jakimanka': 4952, 'Otradnoe': 3949, 'Kuzminki': 3903, 'PoselenieShherbinka': 2189, 'KosinoUhtomskoe': 3419, 'Kurkino': 4004, 'NagatinoSadovniki': 4194, 'Mozhajskoe': 3849, 'Bogorodskoe': 3864, 'Mitino': 4204, 'Zjuzino': 4410, 'Troickijokrug': 2243, 'NagatinskijZaton': 4101, 'PoselenieRogovskoe': 1318, 'SevernoeTushino': 4552, 'Brateevo': 3623, 'StaroeKrjukovo': 2746, 'Krylatskoe': 5255, 'Lianozovo': 3906, 'Konkovo': 4651, 'Zjablikovo': 3895, 'PoselenieMoskovskij': 2522, 'Sokol': 4343, 'Kuncevo': 4325, 'Sviblovo': 4087, 'Silino': 3004, 'Butyrskoe': 4114, 'Rjazanskij': 3797, 'Jasenevo': 4146, 'Gagarinskoe': 5815, 'PoselenieMarushkinskoe': 1841, 'Metrogorodok': 3857, 'PoselenieDesjonovskoe': 2340, 'Cheremushki': 4957, 'Presnenskoe': 5221, 'Marino': 3899, 'Novogireevo': 4064, 'Lefortovo': 4346, 'Goljanovo': 3623, 'Bibirevo': 3812, 'SevernoeButovo': 4021, 'ZapadnoeDegunino': 2820, 'Preobrazhenskoe': 4301, 'Hovrino': 3931, 'Ramenki': 4968, 'PoselenieFilimonkovskoe': 1946, 'Shhukino': 4825, 'ChertanovoCentralnoe': 4177, 'VyhinoZhulebino': 3638, 'Kotlovka': 4412, 'Taganskoe': 4861, 'Tverskoe': 6580, 'TeplyjStan': 4021, 'Perovo': 3739, 'MarinaRoshha': 4588, 'Meshhanskoe': 5676, 'Tekstilshhiki': 3809, 'JuzhnoeTushino': 4051, 'Nagornoe': 4463, 'Solncevo': 3060, 'PoselenieMosrentgen': 2744, 'HoroshevoMnevniki': 4369, 'Krjukovo': 2853, 'BirjulevoVostochnoe': 3195, 'Beskudnikovskoe': 3689, 'OchakovoMatveevskoe': 4047, 'Akademicheskoe': 4692, 'Ivanovskoe': 3505, 'Savelovskoe': 4551, 'SevernoeIzmajlovo': 3886, 'PoselenieKokoshkino': 1956}

    sigma_left = {'Ramenki': 120319.72150269579, 'Cheremushki': 121710.75562529726,
                  'ErrorRaionPoselenie': 66430.692844445643, 'Marfino': 82954.372560305739,
                  'Timirjazevskoe': 96279.543648028557, 'PoselenieKokoshkino': 42815.38160861106,
                  'Krasnoselskoe': 134452.44383364433, 'Vostochnoe': 98438.339322990811, 'Bibirevo': 92347.343670617178,
                  'JuzhnoeTushino': 92616.218419091543, 'OrehovoBorisovoJuzhnoe': 101715.6509342646,
                  'KosinoUhtomskoe': 84191.056341771211, 'ChertanovoSevernoe': 109198.59945730233,
                  'JuzhnoeButovo': 87171.023213745633, 'Gagarinskoe': 138978.29483748571,
                  'Danilovskoe': 110819.64287297914, 'Koptevo': 102331.61513868156,
                  'PoselenieVnukovskoe': 91128.486306592153, 'TeplyjStan': 102268.35471160244,
                  'FiliDavydkovo': 99417.124590694104, 'Babushkinskoe': 97942.009353008005,
                  'Savelovskoe': 111785.1182257686, 'Solncevo': 80771.009080765187,
                  'ChertanovoJuzhnoe': 103077.63820777266, 'Obruchevskoe': 96811.485700936028,
                  'Basmannoe': 124586.39864056923, 'Donskoe': 92070.231052326228, 'Lomonosovskoe': 138237.40695116503,
                  'SevernoeButovo': 108606.05367348727, 'Preobrazhenskoe': 105302.10319179788,
                  'Vojkovskoe': 105219.36843379471, 'Severnoe': 64968.816419512958, 'Hamovniki': 188996.27077075117,
                  'SokolinajaGora': 99874.061061392567, 'StaroeKrjukovo': 57123.750341004881,
                  'Sviblovo': 109928.71394298683, 'PoselenieMihajlovoJarcevskoe': np.NAN, 'Marino': 101933.93683008007,
                  'Dorogomilovo': 146232.71309525287, 'PoselenieRogovskoe': 26627.406068068252,
                  'PoselenieSosenskoe': 75743.744910490437, 'Goljanovo': 83523.341703774582,
                  'Jaroslavskoe': 87851.808415637192, 'Kuncevo': 91994.496178813948, 'Kurkino': 93326.36562775448,
                  'JuzhnoeMedvedkovo': 75253.650371396405, 'Levoberezhnoe': 99491.694933600302,
                  'Krylatskoe': 139829.84745103077, 'Perovo': 92109.756211469299, 'FilevskijPark': 107841.47285033023,
                  'Shhukino': 109627.30398141209, 'Strogino': 119412.95241848381, 'Zjablikovo': 97282.642935924348,
                  'NovoPeredelkino': 80090.957822515877, 'Zjuzino': 107538.13669163339, 'Izmajlovo': 98643.00089737252,
                  'Nizhegorodskoe': 90752.686325341114, 'NagatinoSadovniki': 107577.31745942074,
                  'PoselenieNovofedorovskoe': 49348.899841542487, 'Taganskoe': 95002.493844268436,
                  'Mitino': 118930.99801794275, 'PoselenieMosrentgen': 63323.531524809543,
                  'PoselenieFilimonkovskoe': 59879.721685803728, 'MoskvorecheSaburovo': 105290.71233070534,
                  'Losinoostrovskoe': 107365.33914992982, 'Sokolniki': 137907.47820583929,
                  'SevernoeIzmajlovo': 93126.58584091677, 'PoseleniePervomajskoe': 46805.439951685075,
                  'Alekseevskoe': 117293.10028825593, 'OrehovoBorisovoSevernoe': 87501.171230890846,
                  'Krjukovo': 83425.149298868273, 'NagatinskijZaton': 134741.98011553255, 'Vnukovo': 78491.262669397402,
                  'Jakimanka': 97163.998447735983, 'Beskudnikovskoe': 83219.167879841872,
                  'Troickijokrug': 53486.836032558887, 'Kotlovka': 107849.662150291,
                  'BirjulevoVostochnoe': 79881.861894502319, 'PoselenieMarushkinskoe': 34755.407320300714,
                  'Butyrskoe': 92833.956554632416, 'SevernoeTushino': 115569.64561782315,
                  'Tverskoe': 152226.56870957866, 'TroparevoNikulino': 115795.10138163093,
                  'Lianozovo': 85629.63844860447, 'Konkovo': 119371.61813053247, 'Lefortovo': 101275.91509081508,
                  'Pechatniki': 87923.419294742052, 'PoselenieKrasnopahorskoe': 39071.24360418049,
                  'Zamoskvoreche': 168256.94842642272, 'Golovinskoe': 92331.061574028252,
                  'PoselenieVoskresenskoe': 83085.242555871708, 'MarinaRoshha': 97942.882825297755,
                  'ProspektVernadskogo': 110706.63074592043, 'Caricyno': 103097.13829857926,
                  'PoselenieDesjonovskoe': 79059.1448678897, 'Altufevskoe': 70501.479604822467,
                  'Begovoe': 139016.42108893141, 'OchakovoMatveevskoe': 96955.797332141025,
                  'Tekstilshhiki': 99195.256645138012, 'Ivanovskoe': 82817.560108995967,
                  'Ostankinskoe': 112979.38332071352, 'PoselenieVoronovskoe': 52603.830105900139,
                  'Dmitrovskoe': 92050.014798902659, 'Arbat': 169210.31685631504, 'Ljublino': 95136.623696853931,
                  'Matushkino': 67911.09149439988, 'VostochnoeIzmajlovo': 93285.453481257122,
                  'PoselenieShherbinka': 70158.286178935057, 'ChertanovoCentralnoe': 95750.964580751432,
                  'VyhinoZhulebino': 93137.229967915904, 'Ajeroport': 116105.40994711072,
                  'Novokosino': 103373.55687402253, 'SevernoeMedvedkovo': 94086.737207862316,
                  'Nagornoe': 126679.92705997681, 'Mozhajskoe': 90726.809267938836, 'Hovrino': 99040.925052951337,
                  'Juzhnoportovoe': 117717.1779815146, 'HoroshevoMnevniki': 91746.122928115801,
                  'Otradnoe': 97677.425699321597, 'PokrovskoeStreshnevo': 117762.83644833736,
                  'PoselenieMoskovskij': 78016.556518232333, 'Brateevo': 91813.078380781371,
                  'Silino': 68123.024600405333, 'Nekrasovka': 85959.404354739352, 'Rostokino': 107851.59614968937,
                  'Molzhaninovskoe': 1585.2228165421329, 'Kuzminki': 99385.486662134354,
                  'PoselenieKievskij': 32954.723746149437, 'Jasenevo': 105667.67199509795,
                  'Presnenskoe': 113167.57963421181, 'PoselenieShhapovskoe': 42435.224679802508,
                  'Sokol': 91902.369052812224, 'Akademicheskoe': 109727.8754107093, 'Meshhanskoe': 131464.48940170874,
                  'Novogireevo': 101768.62323211649, 'Savelki': 66852.31965841043, 'Rjazanskij': 94879.7322817042,
                  'Kapotnja': 85068.814345690262, 'Horoshevskoe': 134577.7517284324,
                  'ZapadnoeDegunino': 68626.926683745172, 'PoselenieKlenovskoe': 20000,
                  'BirjulevoZapadnoe': 88793.37587676532, 'Bogorodskoe': 91516.377366515895,
                  'PoselenieRjazanovskoe': 60979.630313594971, 'VostochnoeDegunino': 96881.70936698695,
                  'Metrogorodok': 103629.11405280954, 'Veshnjaki': 87892.408304499026}
    sigma_right = {'Ramenki': 252144.87412329894, 'Cheremushki': 238303.22441795235,
                   'ErrorRaionPoselenie': 156086.68242414642, 'Marfino': 203149.888215399,
                   'Timirjazevskoe': 205995.18700542225, 'PoselenieKokoshkino': 100067.14428618099,
                   'Krasnoselskoe': 255474.97844537883, 'Vostochnoe': 144840.11240857898,
                   'Bibirevo': 181939.14726654542, 'JuzhnoeTushino': 194112.14381724584,
                   'OrehovoBorisovoJuzhnoe': 180660.81013926404, 'KosinoUhtomskoe': 164315.51323220163,
                   'ChertanovoSevernoe': 196139.92635217492, 'JuzhnoeButovo': 167948.62554380979,
                   'Gagarinskoe': 273205.99498550373, 'Danilovskoe': 237964.03185350625, 'Koptevo': 198006.47261959687,
                   'PoselenieVnukovskoe': 130958.77353763449, 'TeplyjStan': 199140.1605912965,
                   'FiliDavydkovo': 225190.99204679206, 'Babushkinskoe': 206892.27207027737,
                   'Savelovskoe': 225110.38062643027, 'Solncevo': 148476.4146772708,
                   'ChertanovoJuzhnoe': 186866.46216614227, 'Obruchevskoe': 229081.14311106424,
                   'Basmannoe': 262879.45898444601, 'Donskoe': 292934.94770770473, 'Lomonosovskoe': 263913.84616634028,
                   'SevernoeButovo': 187914.30149711488, 'Preobrazhenskoe': 200354.81091449709,
                   'Vojkovskoe': 207376.67706410517, 'Severnoe': 155067.62361464722, 'Hamovniki': 366303.12968331482,
                   'SokolinajaGora': 200983.68705538349, 'StaroeKrjukovo': 141820.76932125166,
                   'Sviblovo': 213962.9265659794, 'PoselenieMihajlovoJarcevskoe': np.NAN, 'Marino': 181817.9277698737,
                   'Dorogomilovo': 302139.22080842248, 'PoselenieRogovskoe': 77855.927344525262,
                   'PoselenieSosenskoe': 113249.37947114333, 'Goljanovo': 175934.99184140543,
                   'Jaroslavskoe': 171413.54274987255, 'Kuncevo': 220413.64144380452, 'Kurkino': 199766.03463318531,
                   'JuzhnoeMedvedkovo': 191603.58496202098, 'Levoberezhnoe': 197734.36346276279,
                   'Krylatskoe': 252368.78390747492, 'Perovo': 182845.39718644135, 'FilevskijPark': 230688.99023501808,
                   'Shhukino': 236384.8736298696, 'Strogino': 218255.56010906294, 'Zjablikovo': 185226.2494965326,
                   'NovoPeredelkino': 143058.12152717522, 'Zjuzino': 208077.90500823938, 'Izmajlovo': 199186.9914225509,
                   'Nizhegorodskoe': 176546.64858728321, 'NagatinoSadovniki': 205050.7754514862,
                   'PoselenieNovofedorovskoe': 63333.495739155653, 'Taganskoe': 265494.23494301818,
                   'Mitino': 178847.27009514096, 'PoselenieMosrentgen': 135396.52121074707,
                   'PoselenieFilimonkovskoe': 88808.916325364058, 'MoskvorecheSaburovo': 198944.08984157981,
                   'Losinoostrovskoe': 186960.38986618584, 'Sokolniki': 261898.29052168032,
                   'SevernoeIzmajlovo': 185599.33959413361, 'PoseleniePervomajskoe': 81030.658269484178,
                   'Alekseevskoe': 241562.13340326186, 'OrehovoBorisovoSevernoe': 179621.77611658943,
                   'Krjukovo': 130298.15163361297, 'NagatinskijZaton': 202013.51626428231,
                   'Vnukovo': 137015.58271006882, 'Jakimanka': 265688.45407398569,
                   'Beskudnikovskoe': 187290.66044005577, 'Troickijokrug': 113287.69303724867,
                   'Kotlovka': 211989.64266719358, 'BirjulevoVostochnoe': 156567.62943780649,
                   'PoselenieMarushkinskoe': 106192.11134460925, 'Butyrskoe': 211662.99185500806,
                   'SevernoeTushino': 199042.41418927154, 'Tverskoe': 330013.07477409439,
                   'TroparevoNikulino': 223825.63454995328, 'Lianozovo': 185802.16571639068,
                   'Konkovo': 222191.77463037262, 'Lefortovo': 207950.22979205006, 'Pechatniki': 175992.40535555914,
                   'PoselenieKrasnopahorskoe': 84950.16654828476, 'Zamoskvoreche': 304261.31693231961,
                   'Golovinskoe': 189251.96372360046, 'PoselenieVoskresenskoe': 107517.54180211019,
                   'MarinaRoshha': 231898.04668588017, 'ProspektVernadskogo': 271722.38324594515,
                   'Caricyno': 187490.09906198271, 'PoselenieDesjonovskoe': 104936.88391063726,
                   'Altufevskoe': 177733.17404896111, 'Begovoe': 250127.0903898745,
                   'OchakovoMatveevskoe': 196113.9855512777, 'Tekstilshhiki': 174865.4255997563,
                   'Ivanovskoe': 171636.61400934806, 'Ostankinskoe': 244291.08542434959,
                   'PoselenieVoronovskoe': 93517.403560162798, 'Dmitrovskoe': 170122.6143502401,
                   'Arbat': 370914.96122271288, 'Ljublino': 179197.22422290099, 'Matushkino': 144432.24024995085,
                   'VostochnoeIzmajlovo': 183704.88503859169, 'PoselenieShherbinka': 100678.96146233255,
                   'ChertanovoCentralnoe': 198038.54851818201, 'VyhinoZhulebino': 175880.96895482298,
                   'Ajeroport': 240039.25555805585, 'Novokosino': 172552.62972907664,
                   'SevernoeMedvedkovo': 197721.34661162732, 'Nagornoe': 229222.4419692895,
                   'Mozhajskoe': 196839.19841457723, 'Hovrino': 192624.40689650062,
                   'Juzhnoportovoe': 211623.20514966402, 'HoroshevoMnevniki': 223754.73838630857,
                   'Otradnoe': 192622.30215567711, 'PokrovskoeStreshnevo': 204456.38366904767,
                   'PoselenieMoskovskij': 112810.08710653105, 'Brateevo': 177316.14791398533,
                   'Silino': 144032.22545944288, 'Nekrasovka': 114522.78068729787, 'Rostokino': 207106.97273821643,
                   'Molzhaninovskoe': 77502.496481703478, 'Kuzminki': 180709.34218413034,
                   'PoselenieKievskij': 222508.86966044398, 'Jasenevo': 195857.02440656547,
                   'Presnenskoe': 295774.86966799363, 'PoselenieShhapovskoe': 47636.745017167188,
                   'Sokol': 236188.1712443732, 'Akademicheskoe': 244875.02803596167, 'Meshhanskoe': 276357.91272280819,
                   'Novogireevo': 188112.93826026571, 'Savelki': 146802.48646503664, 'Rjazanskij': 184527.13944099803,
                   'Kapotnja': 142294.54271282727, 'Horoshevskoe': 242868.29285857463,
                   'ZapadnoeDegunino': 135673.67822672089, 'PoselenieKlenovskoe': np.NAN,
                   'BirjulevoZapadnoe': 149206.85123059247, 'Bogorodskoe': 190800.37011261066,
                   'PoselenieRjazanovskoe': 109066.21061835004, 'VostochnoeDegunino': 177701.42221929005,
                   'Metrogorodok': 170598.93199462257, 'Veshnjaki': 169340.62258852244}
    sigma_left2 = {'Ramenki': 54407.145192394208, 'Cheremushki': 63414.521228969723,
                   'ErrorRaionPoselenie': 21602.698054595239, 'Marfino': 22856.614732759102,
                   'Timirjazevskoe': 41421.721969331702, 'PoselenieKokoshkino': 14189.500269826094,
                   'Krasnoselskoe': 73941.17652777709, 'Vostochnoe': 75237.452780196734, 'Bibirevo': 47551.441872653042,
                   'JuzhnoeTushino': 41868.255720014407, 'OrehovoBorisovoJuzhnoe': 62243.071331764892,
                   'KosinoUhtomskoe': 44128.827896556017, 'ChertanovoSevernoe': 65727.936009866055,
                   'JuzhnoeButovo': 46782.222048713564, 'Gagarinskoe': 71864.444763476669,
                   'Danilovskoe': 47247.448382715593, 'Koptevo': 54494.186398223916,
                   'PoselenieVnukovskoe': 71213.342691070982, 'TeplyjStan': 53832.451771755412,
                   'FiliDavydkovo': 36530.190862645133, 'Babushkinskoe': 43466.877994373339,
                   'Savelovskoe': 55122.48702543776, 'Solncevo': 46918.306282512392,
                   'ChertanovoJuzhnoe': 61183.226228587839, 'Obruchevskoe': 30676.656995871919,
                   'Basmannoe': 55439.868468630826, 'Donskoe': -8362.1272753630183, 'Lomonosovskoe': 75399.187343577418,
                   'SevernoeButovo': 68951.929761673469, 'Preobrazhenskoe': 57775.749330448263,
                   'Vojkovskoe': 54140.714118639487, 'Severnoe': 19919.412821945822, 'Hamovniki': 100342.84131446938,
                   'SokolinajaGora': 49319.248064397107, 'StaroeKrjukovo': 14775.240850881499,
                   'Sviblovo': 57911.607631490537, 'PoselenieMihajlovoJarcevskoe': np.NAN, 'Marino': 61991.941360183264,
                   'Dorogomilovo': 68279.459238668089, 'PoselenieRogovskoe': 1013.1454298397439,
                   'PoselenieSosenskoe': 56990.927630163984, 'Goljanovo': 37317.51663495916,
                   'Jaroslavskoe': 46070.941248519521, 'Kuncevo': 27784.923546318663, 'Kurkino': 40106.531125039051,
                   'JuzhnoeMedvedkovo': 17078.683076084126, 'Levoberezhnoe': 50370.36066901905,
                   'Krylatskoe': 83560.379222808682, 'Perovo': 46741.935723983275, 'FilevskijPark': 46417.714157986324,
                   'Shhukino': 46248.519157183342, 'Strogino': 69991.648573194252, 'Zjablikovo': 53310.839655620235,
                   'NovoPeredelkino': 48607.375970186207, 'Zjuzino': 57268.252533330407,
                   'Izmajlovo': 48371.005634783316, 'Nizhegorodskoe': 47855.705194370079,
                   'NagatinoSadovniki': 58840.58846338799, 'PoselenieNovofedorovskoe': 42356.601892735896,
                   'Taganskoe': 9756.6232948935649, 'Mitino': 88972.861979343637,
                   'PoselenieMosrentgen': 27287.036681840778, 'PoselenieFilimonkovskoe': 45415.12436602357,
                   'MoskvorecheSaburovo': 58464.023575268089, 'Losinoostrovskoe': 67567.813791801804,
                   'Sokolniki': 75912.072047918802, 'SevernoeIzmajlovo': 46890.208964308345,
                   'PoseleniePervomajskoe': 29692.83079278552, 'Alekseevskoe': 55158.583730752958,
                   'OrehovoBorisovoSevernoe': 41440.868788041553, 'Krjukovo': 59988.648131495916,
                   'NagatinskijZaton': 101106.21204115768, 'Vnukovo': 49229.102649061693,
                   'Jakimanka': 12901.770634611137, 'Beskudnikovskoe': 31183.421599734909,
                   'Troickijokrug': 23586.407530214005, 'Kotlovka': 55779.671891839695,
                   'BirjulevoVostochnoe': 41538.97812285024, 'PoselenieMarushkinskoe': -962.9446918535541,
                   'Butyrskoe': 33419.4389044446, 'SevernoeTushino': 73833.261332098962, 'Tverskoe': 63333.315677320788,
                   'TroparevoNikulino': 61779.834797469739, 'Lianozovo': 35543.37481471138,
                   'Konkovo': 67961.539880612399, 'Lefortovo': 47938.757740197587, 'Pechatniki': 43888.926264333524,
                   'PoselenieKrasnopahorskoe': 16131.782132128348, 'Zamoskvoreche': 100254.76417347425,
                   'Golovinskoe': 43870.610499242161, 'PoselenieVoskresenskoe': 70869.092932752465,
                   'MarinaRoshha': 30965.300895006541, 'ProspektVernadskogo': 30198.75449590807,
                   'Caricyno': 60900.657916877535, 'PoselenieDesjonovskoe': 66120.275346515918,
                   'Altufevskoe': 16885.63238275316, 'Begovoe': 83461.086438459854,
                   'OchakovoMatveevskoe': 47376.703222572702, 'Tekstilshhiki': 61360.17216782886,
                   'Ivanovskoe': 38408.033158819919, 'Ostankinskoe': 47323.532268895477,
                   'PoselenieVoronovskoe': 32147.043378768802, 'Dmitrovskoe': 53013.715023233934,
                   'Arbat': 68357.994673116133, 'Ljublino': 53106.323433830417, 'Matushkino': 29650.517116624411,
                   'VostochnoeIzmajlovo': 48075.737702589831, 'PoselenieShherbinka': 54897.948537236312,
                   'ChertanovoCentralnoe': 44607.172612036156, 'VyhinoZhulebino': 51765.360474462359,
                   'Ajeroport': 54138.487141638165, 'Novokosino': 68784.020446495473,
                   'SevernoeMedvedkovo': 42269.432505979814, 'Nagornoe': 75408.669605320465,
                   'Mozhajskoe': 37670.614694619624, 'Hovrino': 52249.184131176706,
                   'Juzhnoportovoe': 70764.164397439905, 'HoroshevoMnevniki': 25741.815199019416,
                   'Otradnoe': 50204.987471143831, 'PokrovskoeStreshnevo': 74416.062837982216,
                   'PoselenieMoskovskij': 60619.791224082968, 'Brateevo': 49061.543614179376,
                   'Silino': 30168.424170886559, 'Nekrasovka': 71677.716188460108, 'Rostokino': 58223.907855425859,
                   'Molzhaninovskoe': -36373.414016038536, 'Kuzminki': 58723.558901136363,
                   'PoselenieKievskij': -61822.349210997825, 'Jasenevo': 60572.995789364184,
                   'Presnenskoe': 21863.93461732092, 'PoselenieShhapovskoe': 39834.464511120168,
                   'Sokol': 19759.467957031738, 'Akademicheskoe': 42154.299098083109, 'Meshhanskoe': 59017.777741159021,
                   'Novogireevo': 58596.465718041887, 'Savelki': 26877.236255097319, 'Rjazanskij': 50056.028702057287,
                   'Kapotnja': 56455.950162121764, 'Horoshevskoe': 80432.48116336127,
                   'ZapadnoeDegunino': 35103.550912257298, 'PoselenieKlenovskoe': np.NAN,
                   'BirjulevoZapadnoe': 58586.638199851754, 'Bogorodskoe': 41874.380993468498,
                   'PoselenieRjazanovskoe': 36936.340161217442, 'VostochnoeDegunino': 56471.852940835408,
                   'Metrogorodok': 70144.205081903026, 'Veshnjaki': 47168.301162487318}
    sigma_right2 = {'Ramenki': 318057.45043360052, 'Cheremushki': 296599.45881427987,
                    'ErrorRaionPoselenie': 200914.67721399682, 'Marfino': 263247.64604294568,
                    'Timirjazevskoe': 260853.00868411912, 'PoselenieKokoshkino': 128693.02562496596,
                    'Krasnoselskoe': 315986.24575124605, 'Vostochnoe': 168040.99895137307,
                    'Bibirevo': 226735.04906450957, 'JuzhnoeTushino': 244860.10651632299,
                    'OrehovoBorisovoJuzhnoe': 220133.38974176373, 'KosinoUhtomskoe': 204377.74167741684,
                    'ChertanovoSevernoe': 239610.58979961119, 'JuzhnoeButovo': 208337.42670884184,
                    'Gagarinskoe': 340319.84505951277, 'Danilovskoe': 301536.22634376981, 'Koptevo': 245843.9013600545,
                    'PoselenieVnukovskoe': 150873.91715315566, 'TeplyjStan': 247576.06353114353,
                    'FiliDavydkovo': 288077.92577484099, 'Babushkinskoe': 261367.40342891205,
                    'Savelovskoe': 281773.01182676113, 'Solncevo': 182329.11747552361,
                    'ChertanovoJuzhnoe': 228760.87414532708, 'Obruchevskoe': 295215.97181612835,
                    'Basmannoe': 332025.98915638443, 'Donskoe': 393367.30603539397, 'Lomonosovskoe': 326752.06577392784,
                    'SevernoeButovo': 227568.42540892868, 'Preobrazhenskoe': 247881.1647758467,
                    'Vojkovskoe': 258455.3313792604, 'Severnoe': 200117.02721221437, 'Hamovniki': 454956.55913959665,
                    'SokolinajaGora': 251538.50005237895, 'StaroeKrjukovo': 184169.27881137503,
                    'Sviblovo': 265980.03287747572, 'PoselenieMihajlovoJarcevskoe': np.NAN, 'Marino': 221759.92323977049,
                    'Dorogomilovo': 380092.47466500726, 'PoselenieRogovskoe': 103470.18798275378,
                    'PoselenieSosenskoe': 132002.19675146978, 'Goljanovo': 222140.81691022083,
                    'Jaroslavskoe': 213194.4099169902, 'Kuncevo': 284623.2140762998, 'Kurkino': 252985.86913590075,
                    'JuzhnoeMedvedkovo': 249778.55225733324, 'Levoberezhnoe': 246855.69772734406,
                    'Krylatskoe': 308638.252135697, 'Perovo': 228213.21767392737, 'FilevskijPark': 292112.74892736197,
                    'Shhukino': 299763.65845409833, 'Strogino': 267676.86395435245, 'Zjablikovo': 229198.05277683673,
                    'NovoPeredelkino': 174541.70337950491, 'Zjuzino': 258347.78916654235,
                    'Izmajlovo': 249458.98668514012, 'Nizhegorodskoe': 219443.62971825426,
                    'NagatinoSadovniki': 253787.50444751896, 'PoselenieNovofedorovskoe': 70325.793687962243,
                    'Taganskoe': 350740.10549239302, 'Mitino': 208805.40613374009,
                    'PoselenieMosrentgen': 171433.01605371584, 'PoselenieFilimonkovskoe': 103273.51364514421,
                    'MoskvorecheSaburovo': 245770.77859701705, 'Losinoostrovskoe': 226757.91522431385,
                    'Sokolniki': 323893.69667960081, 'SevernoeIzmajlovo': 231835.71647074205,
                    'PoseleniePervomajskoe': 98143.26742838374, 'Alekseevskoe': 303696.6499607648,
                    'OrehovoBorisovoSevernoe': 225682.07855943873, 'Krjukovo': 153734.65280098532,
                    'NagatinskijZaton': 235649.28433865719, 'Vnukovo': 166277.74273040454,
                    'Jakimanka': 349950.68188711052, 'Beskudnikovskoe': 239326.40672016272,
                    'Troickijokrug': 143188.12153959356, 'Kotlovka': 264059.63292564487,
                    'BirjulevoVostochnoe': 194910.51320945856, 'PoselenieMarushkinskoe': 141910.46335676353,
                    'Butyrskoe': 271077.50950519589, 'SevernoeTushino': 240778.79847499571,
                    'Tverskoe': 418906.32780635229, 'TroparevoNikulino': 277840.90113411448,
                    'Lianozovo': 235888.42935028375, 'Konkovo': 273601.85288029269, 'Lefortovo': 261287.38714266755,
                    'Pechatniki': 220026.89838596765, 'PoselenieKrasnopahorskoe': 107889.62802033691,
                    'Zamoskvoreche': 372263.50118526805, 'Golovinskoe': 237712.41479838657,
                    'PoselenieVoskresenskoe': 119733.69142522944, 'MarinaRoshha': 298875.62861617142,
                    'ProspektVernadskogo': 352230.25949595752, 'Caricyno': 229686.57944368443,
                    'PoselenieDesjonovskoe': 117875.75343201104, 'Altufevskoe': 231349.02127103042,
                    'Begovoe': 305682.42504034605, 'OchakovoMatveevskoe': 245693.07966084604,
                    'Tekstilshhiki': 212700.51007706547, 'Ivanovskoe': 216046.14095952411,
                    'Ostankinskoe': 309946.93647616764, 'PoselenieVoronovskoe': 113974.19028729413,
                    'Dmitrovskoe': 209158.91412590884, 'Arbat': 471767.28340591176, 'Ljublino': 221227.52448592451,
                    'Matushkino': 182692.81462772633, 'VostochnoeIzmajlovo': 228914.60081725899,
                    'PoselenieShherbinka': 115939.29910403129, 'ChertanovoCentralnoe': 249182.34048689727,
                    'VyhinoZhulebino': 217252.83844827654, 'Ajeroport': 302006.17836352839,
                    'Novokosino': 207142.16615660369, 'SevernoeMedvedkovo': 249538.65131350982,
                    'Nagornoe': 280493.69942394586, 'Mozhajskoe': 249895.39298789646, 'Hovrino': 239416.14781827523,
                    'Juzhnoportovoe': 258576.2187337387, 'HoroshevoMnevniki': 289759.04611540493,
                    'Otradnoe': 240094.74038385489, 'PokrovskoeStreshnevo': 247803.15727940283,
                    'PoselenieMoskovskij': 130206.85240068042, 'Brateevo': 220067.68268058734,
                    'Silino': 181986.82588896167, 'Nekrasovka': 128804.46885357711, 'Rostokino': 256734.66103247995,
                    'Molzhaninovskoe': 115461.13331428415, 'Kuzminki': 221371.26994512833,
                    'PoselenieKievskij': 317285.94261759124, 'Jasenevo': 240951.70061229923,
                    'Presnenskoe': 387078.51468488446, 'PoselenieShhapovskoe': 50237.505185849528,
                    'Sokol': 308331.07234015368, 'Akademicheskoe': 312448.60434858786,
                    'Meshhanskoe': 348804.62438335788, 'Novogireevo': 231285.09577434033, 'Savelki': 186777.56986834976,
                    'Rjazanskij': 229350.84302064494, 'Kapotnja': 170907.40689639575,
                    'Horoshevskoe': 297013.56342364574, 'ZapadnoeDegunino': 169197.05399820878,
                    'PoselenieKlenovskoe': np.NAN, 'BirjulevoZapadnoe': 179413.58890750603,
                    'Bogorodskoe': 240442.36648565804, 'PoselenieRjazanovskoe': 133109.50077072758,
                    'VostochnoeDegunino': 218111.27864544158, 'Metrogorodok': 204083.84096552909,
                    'Veshnjaki': 210064.72973053413}

    # raw_data.insert(0, 'min_price_per_raion', np.NAN)
    raw_data.insert(0, 'mean_price_per_raion', np.NAN)
    # raw_data.insert(0, 'max_price_per_raion', np.NAN)
    raw_data.insert(0, 'count_sales_per_raion', np.NAN)
    for k, v in min_price.items():
        raw_data.loc[raw_data['sub_area'] == k, 'min_price_per_raion'] = v

    for k, v in mean_price.items():
        raw_data.loc[raw_data['sub_area'] == k, 'mean_price_per_raion'] = v

    for k, v in max_price.items():
        raw_data.loc[raw_data['sub_area'] == k, 'max_price_per_raion'] = v

    for k, v in count.items():
        raw_data.loc[raw_data['sub_area'] == k, 'count_sales_per_raion'] = v

    # raw_data['sigma_left'] = raw_data['sub_area'].copy().map(sigma_left).astype(float)
    # raw_data['sigma_right'] = raw_data['sub_area'].copy().map(sigma_right).astype(float)
    # raw_data['sigma_left2'] = raw_data['sub_area'].copy().map(sigma_left2).astype(float)
    # raw_data['sigma_right2'] = raw_data['sub_area'].copy().map(sigma_right2).astype(float)


    # print(raw_data['sigma_left'])
    # asd()

    raw_data['sub_area'] = raw_data['sub_area'].map(mean_price_usd).astype(float)
    # raw_data['sub_area'] = np.log(raw_data['sub_area'])
    raw_data.drop(['sub_area'], axis=1, inplace=True)

    # Zelenograd & < -c("Krjukovo", "Matushkino", "Savelki", "Silino", "Staroe Krjukovo")
    # Novomoskovsky < -c("Poselenie Desjonovskoe", "Poselenie Filimonkovskoe", "Poselenie Kokoshkino",
    #                    "Poselenie Marushkinskoe", "Poselenie Moskovskij", "Poselenie Mosrentgen",
    #                    "Poselenie Rjazanovskoe", "Poselenie Shherbinka", "Poselenie Sosenskoe", "Poselenie Vnukovskoe",
    #                    "Poselenie Voskresenskoe")
    # Troitsky < -c("Poselenie Kievskij", "Poselenie Klenovskoe", "Poselenie Krasnopahorskoe",
    #               "Poselenie Mihajlovo-Jarcevskoe", "Poselenie Novofedorovskoe", "Poselenie Pervomajskoe",
    #               "Poselenie Rogovskoe", "Poselenie Shhapovskoe", "Poselenie Voronovskoe", "Troickij okrug")
    # Northern < -c("Ajeroport", "Begovoe", "Beskudnikovskoe", "Dmitrovskoe", "Golovinskoe", "Horoshevskoe", "Hovrino",
    #               "Koptevo", "Levoberezhnoe", "Molzhaninovskoe", "Savelovskoe", "Sokol", "Timirjazevskoe", "Vojkovskoe",
    #               "Vostochnoe Degunino", "Zapadnoe Degunino")
    # Southwest < -c("Akademicheskoe", "Cheremushki", "Gagarinskoe", "Jasenevo", "Juzhnoe Butovo", "Kon'kovo", "Kotlovka",
    #                "Lomonosovskoe", "Obruchevskoe", "Severnoe Butovo", "Teplyj Stan", "Zjuzino")
    # Northeast & lt;
    # -c("Alekseevskoe", "Altuf'evskoe", "Babushkinskoe", "Bibirevo", "Butyrskoe", "Jaroslavskoe", "Juzhnoe Medvedkovo",
    #    "Lianozovo", "Losinoostrovskoe", "Mar'ina Roshha", "Marfino", "Ostankinskoe", "Otradnoe", "Rostokino",
    #    "Severnoe", "Severnoe Medvedkovo", "Sviblovo")
    # Central < -c("Arbat", "Basmannoe", "Hamovniki", "Jakimanka", "Krasnosel'skoe", "Meshhanskoe", "Presnenskoe",
    #              "Taganskoe", "Tverskoe", "Zamoskvorech'e")
    # Southern < -c("Birjulevo Vostochnoe", "Birjulevo Zapadnoe", "Brateevo", "Caricyno", "Chertanovo Central'noe",
    #               "Chertanovo Juzhnoe", "Chertanovo Severnoe", "Danilovskoe", "Donskoe", "Moskvorech'e-Saburovo",
    #               "Nagatino-Sadovniki", "Nagatinskij Zaton", "Nagornoe", "Orehovo-Borisovo Juzhnoe",
    #               "Orehovo-Borisovo Severnoe", "Zjablikovo")
    # Eastern < -c("Bogorodskoe", "Gol'janovo", "Ivanovskoe", "Izmajlovo", "Kosino-Uhtomskoe", "Metrogorodok",
    #              "Novogireevo", "Novokosino", "Perovo", "Preobrazhenskoe", "Severnoe Izmajlovo", "Sokol'niki",
    #              "Sokolinaja Gora", "Veshnjaki", "Vostochnoe", "Vostochnoe Izmajlovo")
    # Western < -c("Dorogomilovo", "Filevskij Park", "Fili Davydkovo", "Krylatskoe", "Kuncevo", "Mozhajskoe",
    #              "Novo-Peredelkino", "Ochakovo-Matveevskoe", "Prospekt Vernadskogo", "Ramenki", "Solncevo",
    #              "Troparevo-Nikulino", "Vnukovo")
    # Northwest < -c("Horoshevo-Mnevniki", "Juzhnoe Tushino", "Kurkino", "Mitino", "Pokrovskoe Streshnevo",
    #                "Severnoe Tushino", "Shhukino", "Strogino")
    # Southeast < -c("Juzhnoportovoe", "Kapotnja", "Kuz'minki", "Lefortovo", "Ljublino", "Mar'ino", "Nekrasovka",
    #                "Nizhegorodskoe", "Pechatniki", "Rjazanskij", "Tekstil'shhiki", "Vyhino-Zhulebino")

    # show_correlations(raw_data, 'sub_area')
    # print(raw_data[np.isnan(raw_data['sub_area'])])
    # asd()

    # print(aggr[['pbm_usd']])
    # print(aggr['pbm_usd'].head())
    # for s in aggr['pbm_usd']:
    #     print(s)

    # show_correlation(raw_data, 'sub_area', 'price_doc')
    # raw_data['sub_area'] = raw_data['sub_area'].cat.codes

    # describe(raw_data, 'sub_area')

    # show_bar_plot(raw_data, 'sub_area', 'price_doc')
    # asd()
    # min_price_usd = {'ZapadnoeDegunino': 432, 'PoselenieKrasnopahorskoe': 413, 'ErrorRaionPoselenie': 814, 'TroparevoNikulino': 409, 'Vojkovskoe': 482, 'Jasenevo': 347, 'PoselenieShhapovskoe': 1194, 'Mitino': 304, 'SevernoeMedvedkovo': 393, 'Solncevo': 368, 'Lianozovo': 444, 'MoskvorecheSaburovo': 270, 'JuzhnoeMedvedkovo': 275, 'Hamovniki': 993, 'Silino': 557, 'JuzhnoeTushino': 281, 'KosinoUhtomskoe': 425, 'Donskoe': 388, 'PoselenieVnukovskoe': 859, 'Kotlovka': 371, 'Gagarinskoe': 531, 'JuzhnoeButovo': 247, 'Hovrino': 611, 'Bibirevo': 350, 'Ivanovskoe': 532, 'Kuncevo': 416, 'Golovinskoe': 370, 'Otradnoe': 305, 'Dorogomilovo': 559, 'Begovoe': 501, 'Strogino': 458, 'PoselenieKievskij': 1727, 'Beskudnikovskoe': 498, 'Vostochnoe': 1997, 'ChertanovoCentralnoe': 396, 'Perovo': 353, 'PoselenieMihajlovoJarcevskoe': 2098, 'Izmajlovo': 466, 'Cheremushki': 320, 'SevernoeButovo': 385, 'Krjukovo': 193, 'PoselenieKokoshkino': 501, 'PoselenieMoskovskij': 801, 'PoselenieDesjonovskoe': 514, 'Marino': 269, 'OrehovoBorisovoSevernoe': 430, 'Preobrazhenskoe': 327, 'Basmannoe': 463, 'Zamoskvoreche': 573, 'Matushkino': 409, 'Akademicheskoe': 335, 'ChertanovoJuzhnoe': 259, 'ChertanovoSevernoe': 381, 'Nagornoe': 484, 'Zjablikovo': 493, 'Rostokino': 452, 'Lomonosovskoe': 522, 'Danilovskoe': 491, 'Altufevskoe': 414, 'Ljublino': 261, 'Alekseevskoe': 390, 'Metrogorodok': 768, 'Jaroslavskoe': 501, 'PoselenieMosrentgen': 901, 'Koptevo': 452, 'SevernoeIzmajlovo': 443, 'Goljanovo': 366, 'Rjazanskij': 492, 'Timirjazevskoe': 407, 'Troickijokrug': 211, 'Savelki': 385, 'Ajeroport': 556, 'Nekrasovka': 362, 'Sokol': 454, 'Pechatniki': 611, 'Bogorodskoe': 451, 'Krylatskoe': 733, 'ProspektVernadskogo': 482, 'PoseleniePervomajskoe': 541, 'Levoberezhnoe': 362, 'StaroeKrjukovo': 361, 'SevernoeTushino': 494, 'Kuzminki': 284, 'OrehovoBorisovoJuzhnoe': 316, 'Losinoostrovskoe': 435, 'Presnenskoe': 412, 'Horoshevskoe': 484, 'Ramenki': 380, 'Butyrskoe': 451, 'Severnoe': 566, 'VostochnoeIzmajlovo': 512, 'BirjulevoZapadnoe': 270, 'Tekstilshhiki': 420, 'Konkovo': 357, 'HoroshevoMnevniki': 342, 'VyhinoZhulebino': 404, 'Kurkino': 469, 'Brateevo': 275, 'Nizhegorodskoe': 320, 'Savelovskoe': 269, 'NovoPeredelkino': 369, 'FiliDavydkovo': 533, 'PoselenieMarushkinskoe': 672, 'NagatinoSadovniki': 521, 'PoselenieShherbinka': 431, 'Shhukino': 405, 'PoselenieRogovskoe': 711, 'Obruchevskoe': 499, 'PoselenieSosenskoe': 518, 'PoselenieFilimonkovskoe': 723, 'TeplyjStan': 350, 'Meshhanskoe': 477, 'Caricyno': 415, 'PoselenieKlenovskoe': 717, 'Dmitrovskoe': 419, 'BirjulevoVostochnoe': 342, 'Krasnoselskoe': 912, 'Vnukovo': 581, 'Kapotnja': 506, 'Marfino': 394, 'OchakovoMatveevskoe': 347, 'Tverskoe': 702, 'Sokolniki': 1080, 'Ostankinskoe': 545, 'Babushkinskoe': 392, 'PokrovskoeStreshnevo': 845, 'Lefortovo': 601, 'NagatinskijZaton': 466, 'MarinaRoshha': 275, 'PoselenieVoskresenskoe': 750, 'Juzhnoportovoe': 390, 'Novogireevo': 490, 'FilevskijPark': 361, 'Molzhaninovskoe': 443, 'Zjuzino': 261, 'VostochnoeDegunino': 370, 'Novokosino': 384, 'SokolinajaGora': 280, 'Jakimanka': 359, 'Arbat': 2010, 'Taganskoe': 202, 'Veshnjaki': 361, 'Mozhajskoe': 489, 'Sviblovo': 487, 'PoselenieVoronovskoe': 996, 'PoselenieNovofedorovskoe': 735, 'PoselenieRjazanovskoe': 420}
    # max_price_usd = {'Ostankinskoe': 8251, 'ChertanovoCentralnoe': 13802, 'TeplyjStan': 7784, 'Vnukovo': 4364, 'PoselenieMoskovskij': 6746, 'Novogireevo': 6642, 'Horoshevskoe': 9809, 'ZapadnoeDegunino': 5678, 'StaroeKrjukovo': 5069, 'PoselenieRogovskoe': 3735, 'SevernoeButovo': 6496, 'OrehovoBorisovoSevernoe': 5956, 'PoseleniePervomajskoe': 3252, 'Presnenskoe': 11545, 'SevernoeIzmajlovo': 7216, 'Izmajlovo': 7839, 'PoselenieMosrentgen': 4098, 'NagatinoSadovniki': 7811, 'Danilovskoe': 7987, 'Savelki': 5126, 'PoselenieVoronovskoe': 2751, 'JuzhnoeTushino': 8905, 'Jasenevo': 6653, 'Juzhnoportovoe': 9268, 'NagatinskijZaton': 6669, 'Cheremushki': 8980, 'Arbat': 12667, 'Krjukovo': 4545, 'Dorogomilovo': 11039, 'VostochnoeDegunino': 5560, 'Losinoostrovskoe': 5830, 'PokrovskoeStreshnevo': 12441, 'Butyrskoe': 7485, 'Lianozovo': 6143, 'Rostokino': 7615, 'HoroshevoMnevniki': 10024, 'JuzhnoeMedvedkovo': 5969, 'Shhukino': 10875, 'Kuncevo': 7927, 'MoskvorecheSaburovo': 7831, 'MarinaRoshha': 15495, 'SokolinajaGora': 7184, 'Dmitrovskoe': 5467, 'Tverskoe': 13687, 'BirjulevoVostochnoe': 6195, 'KosinoUhtomskoe': 5162, 'Obruchevskoe': 11881, 'Beskudnikovskoe': 6322, 'Rjazanskij': 6517, 'Kapotnja': 4943, 'Pechatniki': 5874, 'Lomonosovskoe': 8946, 'Mitino': 7069, 'PoselenieKievskij': 3576, 'FilevskijPark': 7727, 'OrehovoBorisovoJuzhnoe': 6025, 'Timirjazevskoe': 7644, 'Zamoskvoreche': 9993, 'Ramenki': 15518, 'Meshhanskoe': 10023, 'Gagarinskoe': 9849, 'PoselenieMihajlovoJarcevskoe': 2098, 'JuzhnoeButovo': 5944, 'Taganskoe': 11847, 'Vostochnoe': 4631, 'Koptevo': 9571, 'PoselenieKlenovskoe': 717, 'Krasnoselskoe': 9572, 'Basmannoe': 10817, 'SevernoeTushino': 6775, 'PoselenieKrasnopahorskoe': 3192, 'Severnoe': 4886, 'Strogino': 11802, 'Kotlovka': 7734, 'PoselenieShhapovskoe': 1489, 'Bogorodskoe': 10831, 'Babushkinskoe': 10307, 'Zjuzino': 8843, 'Metrogorodok': 5926, 'Alekseevskoe': 9201, 'ErrorRaionPoselenie': 11236, 'Sokolniki': 12001, 'Brateevo': 5637, 'VostochnoeIzmajlovo': 9657, 'Preobrazhenskoe': 6943, 'Jakimanka': 12728, 'Bibirevo': 5794, 'PoselenieVoskresenskoe': 3930, 'OchakovoMatveevskoe': 7422, 'Vojkovskoe': 7097, 'Donskoe': 13434, 'VyhinoZhulebino': 5785, 'PoselenieMarushkinskoe': 2654, 'Novokosino': 5814, 'Begovoe': 8493, 'PoselenieKokoshkino': 3350, 'Hovrino': 7127, 'Molzhaninovskoe': 1374, 'Troickijokrug': 4501, 'Perovo': 6631, 'Mozhajskoe': 7996, 'Sviblovo': 7031, 'Ivanovskoe': 5583, 'PoselenieVnukovskoe': 4362, 'Lefortovo': 8632, 'PoselenieFilimonkovskoe': 3816, 'ChertanovoSevernoe': 7403, 'Altufevskoe': 7461, 'Caricyno': 7196, 'Tekstilshhiki': 10096, 'Akademicheskoe': 9213, 'Nizhegorodskoe': 5857, 'Silino': 6079, 'Veshnjaki': 5584, 'Sokol': 8442, 'Goljanovo': 7783, 'ChertanovoJuzhnoe': 6215, 'Nekrasovka': 6450, 'FiliDavydkovo': 10064, 'Zjablikovo': 5832, 'Nagornoe': 7141, 'PoselenieRjazanovskoe': 3950, 'Kurkino': 6982, 'Konkovo': 7554, 'PoselenieSosenskoe': 11858, 'TroparevoNikulino': 8056, 'Solncevo': 6415, 'Ljublino': 9743, 'Savelovskoe': 7716, 'Marino': 6487, 'Matushkino': 4852, 'Ajeroport': 8703, 'BirjulevoZapadnoe': 4798, 'PoselenieShherbinka': 4295, 'PoselenieNovofedorovskoe': 3365, 'Golovinskoe': 6670, 'Kuzminki': 6688, 'SevernoeMedvedkovo': 6575, 'Krylatskoe': 10345, 'PoselenieDesjonovskoe': 3211, 'Hamovniki': 14935, 'ProspektVernadskogo': 16650, 'Otradnoe': 6931, 'Levoberezhnoe': 7149, 'Jaroslavskoe': 5290, 'NovoPeredelkino': 5025, 'Marfino': 6507}

    # mean_price_usd = {'SevernoeMedvedkovo': 3895, 'PoselenieKokoshkino': 1956, 'OrehovoBorisovoJuzhnoe': 3949, 'VyhinoZhulebino': 3638, 'Mitino': 4204, 'Timirjazevskoe': 4214, 'Veshnjaki': 3531, 'Novokosino': 3708, 'Ajeroport': 4903, 'Metrogorodok': 3857, 'PoselenieNovofedorovskoe': 1238, 'Akademicheskoe': 4692, 'NagatinskijZaton': 4101, 'Caricyno': 4029, 'Mozhajskoe': 3849, 'SevernoeTushino': 4552, 'ChertanovoCentralnoe': 4177, 'Kuzminki': 3903, 'Dmitrovskoe': 3636, 'PoselenieFilimonkovskoe': 1946, 'Koptevo': 4167, 'PokrovskoeStreshnevo': 4546, 'JuzhnoeTushino': 4051, 'PoselenieMihajlovoJarcevskoe': 2098, 'Krylatskoe': 5255, 'HoroshevoMnevniki': 4369, 'PoselenieKrasnopahorskoe': 1785, 'Horoshevskoe': 4905, 'PoselenieShhapovskoe': 1341, 'Nagornoe': 4463, 'Savelki': 2957, 'Meshhanskoe': 5676, 'PoselenieDesjonovskoe': 2340, 'Lefortovo': 4346, 'ZapadnoeDegunino': 2820, 'Rostokino': 4115, 'NovoPeredelkino': 3179, 'Lianozovo': 3906, 'TeplyjStan': 4021, 'NagatinoSadovniki': 4194, 'KosinoUhtomskoe': 3419, 'Matushkino': 2868, 'Zamoskvoreche': 6480, 'Vostochnoe': 3512, 'PoselenieMosrentgen': 2744, 'Ostankinskoe': 4855, 'Obruchevskoe': 4702, 'ErrorRaionPoselenie': 2848, 'Cheremushki': 4957, 'Taganskoe': 4861, 'FilevskijPark': 4214, 'Troickijokrug': 2243, 'PoseleniePervomajskoe': 1681, 'Silino': 3004, 'Perovo': 3739, 'MoskvorecheSaburovo': 4170, 'Hamovniki': 7796, 'Novogireevo': 4064, 'BirjulevoVostochnoe': 3195, 'Jaroslavskoe': 3521, 'Nekrasovka': 2857, 'Nizhegorodskoe': 3753, 'Beskudnikovskoe': 3689, 'PoselenieKievskij': 2651, 'Rjazanskij': 3797, 'SokolinajaGora': 4131, 'Krasnoselskoe': 5726, 'Severnoe': 3063, 'VostochnoeIzmajlovo': 3734, 'Pechatniki': 3697, 'Jasenevo': 4146, 'Golovinskoe': 3821, 'PoselenieVnukovskoe': 2737, 'Dorogomilovo': 6129, 'Jakimanka': 4952, 'Zjablikovo': 3895, 'Shhukino': 4825, 'Altufevskoe': 3532, 'Goljanovo': 3623, 'Sviblovo': 4087, 'Strogino': 4781, 'MarinaRoshha': 4588, 'JuzhnoeButovo': 3468, 'Marino': 3899, 'Krjukovo': 2853, 'Ljublino': 3788, 'BirjulevoZapadnoe': 3327, 'JuzhnoeMedvedkovo': 3664, 'Zjuzino': 4410, 'Levoberezhnoe': 3752, 'FiliDavydkovo': 4403, 'Presnenskoe': 5221, 'Izmajlovo': 4139, 'Ramenki': 4968, 'PoselenieKlenovskoe': 717, 'Butyrskoe': 4114, 'Marfino': 3929, 'Solncevo': 3060, 'SevernoeIzmajlovo': 3886, 'Danilovskoe': 4558, 'Alekseevskoe': 4751, 'PoselenieVoronovskoe': 2024, 'Brateevo': 3623, 'Preobrazhenskoe': 4301, 'Ivanovskoe': 3505, 'Tekstilshhiki': 3809, 'Vojkovskoe': 4343, 'VostochnoeDegunino': 3748, 'Juzhnoportovoe': 4670, 'PoselenieVoskresenskoe': 2852, 'PoselenieMoskovskij': 2522, 'SevernoeButovo': 4021, 'Otradnoe': 3949, 'Lomonosovskoe': 5427, 'Kotlovka': 4412, 'ProspektVernadskogo': 5350, 'Molzhaninovskoe': 763, 'Tverskoe': 6580, 'Konkovo': 4651, 'Begovoe': 5296, 'Kapotnja': 3342, 'Savelovskoe': 4551, 'Babushkinskoe': 4340, 'PoselenieMarushkinskoe': 1841, 'PoselenieSosenskoe': 2591, 'OchakovoMatveevskoe': 4047, 'Arbat': 7770, 'OrehovoBorisovoSevernoe': 3612, 'Bibirevo': 3812, 'Losinoostrovskoe': 3963, 'Bogorodskoe': 3864, 'TroparevoNikulino': 4560, 'PoselenieRjazanovskoe': 2161, 'ChertanovoJuzhnoe': 4020, 'Kuncevo': 4325, 'Kurkino': 4004, 'Donskoe': 5264, 'Gagarinskoe': 5815, 'Vnukovo': 3139, 'StaroeKrjukovo': 2746, 'ChertanovoSevernoe': 4178, 'Basmannoe': 5369, 'Sokol': 4343, 'Sokolniki': 5700, 'Hovrino': 3931, 'PoselenieRogovskoe': 1318, 'PoselenieShherbinka': 2189}
    # min_price_usd = {'JuzhnoeTushino': 281, 'Novokosino': 384, 'Dorogomilovo': 559, 'PoselenieMosrentgen': 901, 'VostochnoeIzmajlovo': 512, 'Mitino': 304, 'Hovrino': 611, 'PoselenieKlenovskoe': 717, 'Zjablikovo': 493, 'KosinoUhtomskoe': 425, 'PoselenieShhapovskoe': 1194, 'PoselenieMarushkinskoe': 672, 'PoselenieKievskij': 1727, 'Babushkinskoe': 392, 'Matushkino': 409, 'Vnukovo': 581, 'TeplyjStan': 350, 'Jaroslavskoe': 501, 'Danilovskoe': 491, 'Jasenevo': 347, 'Lianozovo': 444, 'Alekseevskoe': 390, 'Metrogorodok': 768, 'ErrorRaionPoselenie': 814, 'Koptevo': 452, 'NagatinskijZaton': 466, 'SevernoeIzmajlovo': 443, 'NagatinoSadovniki': 521, 'Ostankinskoe': 545, 'Strogino': 458, 'NovoPeredelkino': 369, 'Kurkino': 469, 'Ivanovskoe': 532, 'Akademicheskoe': 335, 'Obruchevskoe': 499, 'Otradnoe': 305, 'Pechatniki': 611, 'OrehovoBorisovoSevernoe': 430, 'Sokol': 454, 'Juzhnoportovoe': 390, 'Perovo': 353, 'Savelki': 385, 'PoselenieNovofedorovskoe': 735, 'Arbat': 2010, 'Savelovskoe': 269, 'BirjulevoZapadnoe': 270, 'PokrovskoeStreshnevo': 845, 'FilevskijPark': 361, 'Novogireevo': 490, 'Kuncevo': 416, 'Brateevo': 275, 'Silino': 557, 'Konkovo': 357, 'Timirjazevskoe': 407, 'SevernoeTushino': 494, 'Losinoostrovskoe': 435, 'Zjuzino': 261, 'Bogorodskoe': 451, 'Shhukino': 405, 'HoroshevoMnevniki': 342, 'SevernoeButovo': 385, 'Nizhegorodskoe': 320, 'Lefortovo': 601, 'FiliDavydkovo': 533, 'PoselenieKokoshkino': 501, 'Vojkovskoe': 482, 'Donskoe': 388, 'JuzhnoeButovo': 247, 'MarinaRoshha': 275, 'ProspektVernadskogo': 482, 'Rostokino': 452, 'PoselenieSosenskoe': 518, 'PoselenieRogovskoe': 711, 'Veshnjaki': 361, 'Meshhanskoe': 477, 'Presnenskoe': 412, 'Golovinskoe': 370, 'TroparevoNikulino': 409, 'Hamovniki': 993, 'Caricyno': 415, 'Preobrazhenskoe': 327, 'Rjazanskij': 492, 'StaroeKrjukovo': 361, 'Beskudnikovskoe': 498, 'PoselenieRjazanovskoe': 420, 'PoselenieVoskresenskoe': 750, 'JuzhnoeMedvedkovo': 275, 'Marfino': 394, 'VostochnoeDegunino': 370, 'Kuzminki': 284, 'PoselenieVnukovskoe': 859, 'Sviblovo': 487, 'SokolinajaGora': 280, 'Severnoe': 566, 'PoselenieShherbinka': 431, 'VyhinoZhulebino': 404, 'Lomonosovskoe': 522, 'Troickijokrug': 211, 'Izmajlovo': 466, 'Ajeroport': 556, 'Krjukovo': 193, 'MoskvorecheSaburovo': 270, 'Sokolniki': 1080, 'SevernoeMedvedkovo': 393, 'Butyrskoe': 451, 'PoseleniePervomajskoe': 541, 'PoselenieMoskovskij': 801, 'Cheremushki': 320, 'Gagarinskoe': 531, 'Levoberezhnoe': 362, 'ZapadnoeDegunino': 432, 'ChertanovoSevernoe': 381, 'Vostochnoe': 1997, 'PoselenieVoronovskoe': 996, 'Marino': 269, 'PoselenieMihajlovoJarcevskoe': 2098, 'Kapotnja': 506, 'Altufevskoe': 414, 'Dmitrovskoe': 419, 'Mozhajskoe': 489, 'Zamoskvoreche': 573, 'Horoshevskoe': 484, 'Nekrasovka': 362, 'ChertanovoJuzhnoe': 259, 'Bibirevo': 350, 'Tverskoe': 702, 'Krylatskoe': 733, 'BirjulevoVostochnoe': 342, 'Goljanovo': 366, 'Krasnoselskoe': 912, 'PoselenieKrasnopahorskoe': 413, 'PoselenieDesjonovskoe': 514, 'Basmannoe': 463, 'OchakovoMatveevskoe': 347, 'Tekstilshhiki': 420, 'Nagornoe': 484, 'Begovoe': 501, 'Ramenki': 380, 'Ljublino': 261, 'Molzhaninovskoe': 443, 'Taganskoe': 202, 'Solncevo': 368, 'Kotlovka': 371, 'Jakimanka': 359, 'PoselenieFilimonkovskoe': 723, 'ChertanovoCentralnoe': 396, 'OrehovoBorisovoJuzhnoe': 316}
    # max_price_usd = {'Savelovskoe': 7716, 'Dmitrovskoe': 5467, 'Bogorodskoe': 10831, 'ProspektVernadskogo': 16650, 'Krasnoselskoe': 9572, 'Vostochnoe': 4631, 'Konkovo': 7554, 'Koptevo': 9571, 'SokolinajaGora': 7184, 'PoselenieVoronovskoe': 2751, 'PoselenieFilimonkovskoe': 3816, 'Krjukovo': 4545, 'OrehovoBorisovoSevernoe': 5956, 'BirjulevoVostochnoe': 6195, 'PoselenieKokoshkino': 3350, 'Bibirevo': 5794, 'Matushkino': 4852, 'Severnoe': 4886, 'ZapadnoeDegunino': 5678, 'Beskudnikovskoe': 6322, 'Hamovniki': 14935, 'Sokol': 8442, 'Losinoostrovskoe': 5830, 'Altufevskoe': 7461, 'PoselenieMihajlovoJarcevskoe': 2098, 'PoselenieDesjonovskoe': 3211, 'PoseleniePervomajskoe': 3252, 'Silino': 6079, 'Mitino': 7069, 'PokrovskoeStreshnevo': 12441, 'Molzhaninovskoe': 1374, 'Novokosino': 5814, 'PoselenieVoskresenskoe': 3930, 'Jaroslavskoe': 5290, 'JuzhnoeButovo': 5944, 'Ljublino': 9743, 'KosinoUhtomskoe': 5162, 'SevernoeMedvedkovo': 6575, 'Rostokino': 7615, 'TroparevoNikulino': 8056, 'Veshnjaki': 5584, 'Troickijokrug': 4501, 'Dorogomilovo': 11039, 'NagatinoSadovniki': 7811, 'Marino': 6487, 'VostochnoeIzmajlovo': 9657, 'Mozhajskoe': 7996, 'PoselenieMarushkinskoe': 2654, 'Arbat': 12667, 'Presnenskoe': 11545, 'SevernoeButovo': 6496, 'Zamoskvoreche': 9993, 'Nagornoe': 7141, 'Perovo': 6631, 'Ivanovskoe': 5583, 'NovoPeredelkino': 5025, 'ChertanovoSevernoe': 7403, 'Kapotnja': 4943, 'Cheremushki': 8980, 'VyhinoZhulebino': 5785, 'PoselenieMosrentgen': 4098, 'Jasenevo': 6653, 'StaroeKrjukovo': 5069, 'TeplyjStan': 7784, 'MoskvorecheSaburovo': 7831, 'Babushkinskoe': 10307, 'Brateevo': 5637, 'PoselenieKievskij': 3576, 'Novogireevo': 6642, 'SevernoeTushino': 6775, 'Lomonosovskoe': 8946, 'Basmannoe': 10817, 'PoselenieRjazanovskoe': 3950, 'Goljanovo': 7783, 'Sokolniki': 12001, 'Zjablikovo': 5832, 'Ajeroport': 8703, 'Preobrazhenskoe': 6943, 'Levoberezhnoe': 7149, 'HoroshevoMnevniki': 10024, 'Nekrasovka': 6450, 'Rjazanskij': 6517, 'Zjuzino': 8843, 'Meshhanskoe': 10023, 'FiliDavydkovo': 10064, 'ChertanovoJuzhnoe': 6215, 'Krylatskoe': 10345, 'Horoshevskoe': 9809, 'PoselenieNovofedorovskoe': 3365, 'Lianozovo': 6143, 'PoselenieRogovskoe': 3735, 'Hovrino': 7127, 'JuzhnoeTushino': 8905, 'Izmajlovo': 7839, 'Metrogorodok': 5926, 'Nizhegorodskoe': 5857, 'Otradnoe': 6931, 'JuzhnoeMedvedkovo': 5969, 'Donskoe': 13434, 'Taganskoe': 11847, 'PoselenieShhapovskoe': 1489, 'Sviblovo': 7031, 'MarinaRoshha': 15495, 'Kuncevo': 7927, 'Timirjazevskoe': 7644, 'OchakovoMatveevskoe': 7422, 'Kuzminki': 6688, 'Strogino': 11802, 'PoselenieShherbinka': 4295, 'PoselenieKlenovskoe': 717, 'Pechatniki': 5874, 'Marfino': 6507, 'Gagarinskoe': 9849, 'Jakimanka': 12728, 'Vnukovo': 4364, 'Akademicheskoe': 9213, 'PoselenieVnukovskoe': 4362, 'Tverskoe': 13687, 'Obruchevskoe': 11881, 'Savelki': 5126, 'SevernoeIzmajlovo': 7216, 'PoselenieMoskovskij': 6746, 'Tekstilshhiki': 10096, 'Ostankinskoe': 8251, 'Shhukino': 10875, 'ErrorRaionPoselenie': 11236, 'OrehovoBorisovoJuzhnoe': 6025, 'Kotlovka': 7734, 'Caricyno': 7196, 'NagatinskijZaton': 6669, 'VostochnoeDegunino': 5560, 'PoselenieKrasnopahorskoe': 3192, 'Golovinskoe': 6670, 'BirjulevoZapadnoe': 4798, 'ChertanovoCentralnoe': 13802, 'Juzhnoportovoe': 9268, 'Begovoe': 8493, 'Lefortovo': 8632, 'Ramenki': 15518, 'Vojkovskoe': 7097, 'Solncevo': 6415, 'Kurkino': 6982, 'Danilovskoe': 7987, 'Butyrskoe': 7485, 'PoselenieSosenskoe': 11858, 'Alekseevskoe': 9201, 'FilevskijPark': 7727}

    # mean_price = {'Krasnoselskoe': 194963, 'Alekseevskoe': 177749, 'Vostochnoe': 121639, 'TroparevoNikulino': 168554, 'Preobrazhenskoe': 151910, 'Kuzminki': 139478, 'Mozhajskoe': 143783, 'Kotlovka': 159919, 'HoroshevoMnevniki': 157191, 'VostochnoeDegunino': 136241, 'Gagarinskoe': 206092, 'JuzhnoeTushino': 143364, 'Metrogorodok': 137114, 'NagatinskijZaton': 168377, 'SevernoeMedvedkovo': 145109, 'PoseleniePervomajskoe': 63918, 'Sokolniki': 199902, 'PoselenieKokoshkino': 71441, 'PoselenieNovofedorovskoe': 56341, 'Kuncevo': 156204, 'Zjuzino': 157247, 'Butyrskoe': 150873, 'Jaroslavskoe': 129632, 'SokolinajaGora': 149693, 'Akademicheskoe': 176540, 'Bogorodskoe': 140742, 'Levoberezhnoe': 147598, 'JuzhnoeButovo': 127306, 'KosinoUhtomskoe': 123312, 'Begovoe': 194571, 'Veshnjaki': 128616, 'Horoshevskoe': 187425, 'Nagornoe': 177951, 'MoskvorecheSaburovo': 150705, 'Altufevskoe': 122475, 'Tekstilshhiki': 136615, 'PoselenieMarushkinskoe': 70473, 'Molzhaninovskoe': 39543, 'Rjazanskij': 139703, 'OrehovoBorisovoJuzhnoe': 141188, 'PoselenieRjazanovskoe': 85022, 'PoselenieRogovskoe': 52241, 'Jasenevo': 150189, 'Lianozovo': 135715, 'StaroeKrjukovo': 98523, 'MarinaRoshha': 163610, 'SevernoeTushino': 157306, 'Ljublino': 136744, 'Zjablikovo': 141254, 'VostochnoeIzmajlovo': 138495, 'PoselenieShhapovskoe': 45035, 'SevernoeIzmajlovo': 139362, 'Shhukino': 171975, 'PoselenieMoskovskij': 95413, 'Losinoostrovskoe': 146407, 'PoselenieMosrentgen': 99360, 'Savelki': 105914, 'ZapadnoeDegunino': 101933, 'PoselenieMihajlovoJarcevskoe': 72500, 'Lomonosovskoe': 201075, 'Pechatniki': 131957, 'ErrorRaionPoselenie': 111258, 'Izmajlovo': 148914, 'PoselenieKlenovskoe': 23255, 'Konkovo': 170781, 'Hovrino': 145832, 'PoselenieFilimonkovskoe': 74344, 'PokrovskoeStreshnevo': 161109, 'Meshhanskoe': 203911, 'Silino': 106077, 'Beskudnikovskoe': 135254, 'PoselenieDesjonovskoe': 91998, 'Brateevo': 133901, 'PoselenieShherbinka': 85418, 'Dmitrovskoe': 131086, 'Sokol': 164045, 'Kurkino': 146546, 'Ramenki': 184088, 'PoselenieVnukovskoe': 111043, 'BirjulevoVostochnoe': 118224, 'PoselenieVoskresenskoe': 95301, 'Savelovskoe': 168447, 'NagatinoSadovniki': 156314, 'Presnenskoe': 203466, 'Babushkinskoe': 152417, 'Basmannoe': 193732, 'Bibirevo': 137143, 'Cheremushki': 180006, 'PoselenieSosenskoe': 94496, 'Donskoe': 191171, 'Severnoe': 110018, 'ChertanovoJuzhnoe': 144005, 'ProspektVernadskogo': 191214, 'FiliDavydkovo': 162304, 'Novokosino': 137061, 'Goljanovo': 129329, 'Vnukovo': 107753, 'NovoPeredelkino': 109627, 'Zamoskvoreche': 236259, 'ChertanovoSevernoe': 152669, 'Vojkovskoe': 156298, 'Ostankinskoe': 178635, 'Nizhegorodskoe': 133649, 'PoselenieKievskij': 127731, 'Krjukovo': 105430, 'PoselenieKrasnopahorskoe': 62010, 'Strogino': 168834, 'Arbat': 270062, 'Sviblovo': 161945, 'OrehovoBorisovoSevernoe': 132978, 'Koptevo': 149507, 'Dorogomilovo': 224185, 'Juzhnoportovoe': 164670, 'FilevskijPark': 169265, 'Troickijokrug': 82940, 'Krylatskoe': 196099, 'Novogireevo': 144940, 'Golovinskoe': 139659, 'Marino': 141365, 'TeplyjStan': 149872, 'Nekrasovka': 100088, 'Rostokino': 153140, 'Ivanovskoe': 127227, 'Hamovniki': 277649, 'Perovo': 136971, 'JuzhnoeMedvedkovo': 131759, 'PoselenieVoronovskoe': 73060, 'Matushkino': 103663, 'Caricyno': 145293, 'BirjulevoZapadnoe': 118093, 'Taganskoe': 179282, 'ChertanovoCentralnoe': 146894, 'SevernoeButovo': 146767, 'Kapotnja': 113681, 'Marfino': 141521, 'Mitino': 148689, 'Tverskoe': 241119, 'Lefortovo': 154613, 'Obruchevskoe': 162946, 'Danilovskoe': 174391, 'Ajeroport': 178072, 'Otradnoe': 144044, 'Timirjazevskoe': 151137, 'Solncevo': 114383, 'VyhinoZhulebino': 134509, 'OchakovoMatveevskoe': 146012, 'Jakimanka': 177261}
    # min_price = {'Kotlovka': 17241, 'FiliDavydkovo': 17155, 'Jakimanka': 12500, 'Mozhajskoe': 20408, 'PoselenieDesjonovskoe': 17678, 'Savelovskoe': 17543, 'Kuncevo': 16438, 'Babushkinskoe': 18867, 'Zjablikovo': 16666, 'Ajeroport': 25000, 'PoselenieVoskresenskoe': 26913, 'Losinoostrovskoe': 13513, 'PoselenieKrasnopahorskoe': 16393, 'Alekseevskoe': 13333, 'SokolinajaGora': 12226, 'Danilovskoe': 17241, 'Tverskoe': 27777, 'Goljanovo': 12658, 'Konkovo': 19411, 'PoselenieVnukovskoe': 44880, 'Akademicheskoe': 14558, 'Severnoe': 18518, 'Marfino': 12987, 'Bibirevo': 16393, 'Horoshevskoe': 14866, 'Zjuzino': 12060, 'Jasenevo': 14925, 'NovoPeredelkino': 13333, 'Presnenskoe': 14492, 'Vojkovskoe': 15625, 'Veshnjaki': 16750, 'ProspektVernadskogo': 22000, 'PoselenieVoronovskoe': 42803, 'Gagarinskoe': 28571, 'VyhinoZhulebino': 16779, 'PoselenieFilimonkovskoe': 23571, 'Ramenki': 13513, 'Perovo': 12987, 'Kapotnja': 17460, 'JuzhnoeTushino': 16129, 'NagatinskijZaton': 15151, 'PoselenieMarushkinskoe': 21739, 'Izmajlovo': 16129, 'ZapadnoeDegunino': 13513, 'TeplyjStan': 14271, 'Molzhaninovskoe': 16000, 'Lefortovo': 21818, 'MoskvorecheSaburovo': 12345, 'Nekrasovka': 12099, 'Krjukovo': 12487, 'Zamoskvoreche': 20408, 'Ljublino': 12048, 'OchakovoMatveevskoe': 13414, 'OrehovoBorisovoJuzhnoe': 16666, 'Beskudnikovskoe': 15873, 'StaroeKrjukovo': 13157, 'Begovoe': 15151, 'Jaroslavskoe': 16650, 'Golovinskoe': 13333, 'Sokolniki': 36363, 'Vostochnoe': 82222, 'NagatinoSadovniki': 18867, 'Obruchevskoe': 16666, 'ErrorRaionPoselenie': 29478, 'Sviblovo': 17543, 'Strogino': 15625, 'Taganskoe': 13157, 'PoselenieShhapovskoe': 43196, 'PokrovskoeStreshnevo': 29411, 'PoselenieMoskovskij': 40000, 'Preobrazhenskoe': 14285, 'Dmitrovskoe': 16129, 'Altufevskoe': 12500, 'SevernoeButovo': 12820, 'Novogireevo': 17857, 'Hovrino': 21276, 'Kurkino': 25641, 'Juzhnoportovoe': 22200, 'Tekstilshhiki': 13513, 'Krylatskoe': 26315, 'VostochnoeIzmajlovo': 20833, 'Silino': 19230, 'SevernoeMedvedkovo': 13157, 'Marino': 12500, 'Koptevo': 13888, 'Nagornoe': 19607, 'Arbat': 91666, 'TroparevoNikulino': 12820, 'HoroshevoMnevniki': 12345, 'Brateevo': 13888, 'JuzhnoeMedvedkovo': 13200, 'Rostokino': 14285, 'Novokosino': 13513, 'Butyrskoe': 13333, 'Solncevo': 13888, 'BirjulevoZapadnoe': 14705, 'Cheremushki': 18867, 'FilevskijPark': 17068, 'PoselenieRogovskoe': 39925, 'PoselenieKievskij': 60714, 'Bogorodskoe': 14705, 'Nizhegorodskoe': 17857, 'PoseleniePervomajskoe': 21532, 'Troickijokrug': 12345, 'Krasnoselskoe': 28571, 'MarinaRoshha': 12987, 'Mitino': 13888, 'PoselenieKlenovskoe': 23255, 'PoselenieRjazanovskoe': 22000, 'KosinoUhtomskoe': 12531, 'PoselenieNovofedorovskoe': 39796, 'Rjazanskij': 15384, 'SevernoeIzmajlovo': 17678, 'Dorogomilovo': 17543, 'PoselenieMihajlovoJarcevskoe': 72500, 'Ivanovskoe': 16949, 'Sokol': 16129, 'Timirjazevskoe': 17857, 'Savelki': 13750, 'Otradnoe': 12692, 'Lianozovo': 15873, 'Donskoe': 12820, 'Hamovniki': 42553, 'Vnukovo': 18333, 'Kuzminki': 14285, 'SevernoeTushino': 16666, 'ChertanovoSevernoe': 22727, 'PoselenieSosenskoe': 20833, 'PoselenieMosrentgen': 30937, 'Shhukino': 13200, 'Meshhanskoe': 15147, 'Basmannoe': 15000, 'Matushkino': 12820, 'VostochnoeDegunino': 13333, 'BirjulevoVostochnoe': 15151, 'Ostankinskoe': 17543, 'Pechatniki': 19411, 'Caricyno': 19607, 'Metrogorodok': 27027, 'OrehovoBorisovoSevernoe': 13378, 'Levoberezhnoe': 12658, 'PoselenieKokoshkino': 15625, 'PoselenieShherbinka': 21505, 'Lomonosovskoe': 16229, 'ChertanovoJuzhnoe': 13157, 'JuzhnoeButovo': 13513, 'ChertanovoCentralnoe': 15468}
    # max_price = {'Bogorodskoe': 314102, 'Ostankinskoe': 313750, 'Vnukovo': 153846, 'Lianozovo': 206944, 'Akademicheskoe': 300000, 'Jaroslavskoe': 207662, 'ZapadnoeDegunino': 180000, 'VostochnoeDegunino': 208333, 'Novokosino': 182191, 'Savelki': 168750, 'Timirjazevskoe': 268867, 'NagatinoSadovniki': 256756, 'Izmajlovo': 279691, 'Hamovniki': 541666, 'ChertanovoJuzhnoe': 209103, 'PoselenieVoronovskoe': 96725, 'Hovrino': 225925, 'Sokol': 292272, 'PoselenieMosrentgen': 148571, 'Novogireevo': 206140, 'Rostokino': 272463, 'Begovoe': 312195, 'PoselenieShherbinka': 151724, 'Krasnoselskoe': 312500, 'PoselenieNovofedorovskoe': 103448, 'Metrogorodok': 194444, 'PoselenieKlenovskoe': 23255, 'Goljanovo': 300000, 'Meshhanskoe': 382258, 'HoroshevoMnevniki': 320000, 'NagatinskijZaton': 233823, 'Babushkinskoe': 343155, 'Caricyno': 264705, 'Vojkovskoe': 234328, 'Lomonosovskoe': 323394, 'Arbat': 451612, 'Rjazanskij': 214285, 'Savelovskoe': 263566, 'SevernoeMedvedkovo': 219047, 'OrehovoBorisovoSevernoe': 212121, 'Horoshevskoe': 334408, 'Ajeroport': 471851, 'Mozhajskoe': 274509, 'Krjukovo': 155555, 'VostochnoeIzmajlovo': 292307, 'FilevskijPark': 274000, 'PoselenieMihajlovoJarcevskoe': 72500, 'Butyrskoe': 273437, 'Zjablikovo': 192105, 'Severnoe': 167088, 'MarinaRoshha': 490445, 'Lefortovo': 279166, 'Otradnoe': 230519, 'Troickijokrug': 141666, 'Strogino': 364431, 'Nizhegorodskoe': 208333, 'PoselenieKokoshkino': 122580, 'Sviblovo': 236363, 'Bibirevo': 191891, 'PoselenieSosenskoe': 399826, 'Mitino': 247563, 'SevernoeButovo': 223684, 'TeplyjStan': 237735, 'Ivanovskoe': 200000, 'Donskoe': 532545, 'Altufevskoe': 229411, 'ProspektVernadskogo': 539393, 'Kuncevo': 368595, 'ChertanovoSevernoe': 227631, 'SevernoeTushino': 222017, 'JuzhnoeButovo': 193750, 'MoskvorecheSaburovo': 243589, 'Pechatniki': 193750, 'Taganskoe': 425531, 'Zjuzino': 285714, 'Krylatskoe': 321739, 'PoselenieShhapovskoe': 46875, 'Preobrazhenskoe': 225806, 'Kurkino': 230000, 'OchakovoMatveevskoe': 232456, 'Kuzminki': 255319, 'Nekrasovka': 232894, 'SevernoeIzmajlovo': 223809, 'VyhinoZhulebino': 196710, 'Juzhnoportovoe': 290740, 'Jasenevo': 228947, 'Tekstilshhiki': 325581, 'PoselenieKievskij': 194749, 'JuzhnoeMedvedkovo': 210256, 'PokrovskoeStreshnevo': 382535, 'Presnenskoe': 432374, 'PoselenieVnukovskoe': 170454, 'PoseleniePervomajskoe': 105882, 'Gagarinskoe': 349206, 'KosinoUhtomskoe': 173684, 'Jakimanka': 379032, 'BirjulevoVostochnoe': 205000, 'Ramenki': 492249, 'Ljublino': 288888, 'PoselenieFilimonkovskoe': 137790, 'PoselenieDesjonovskoe': 125491, 'OrehovoBorisovoJuzhnoe': 200000, 'Kapotnja': 168000, 'Koptevo': 297297, 'Cheremushki': 304878, 'Golovinskoe': 216666, 'Brateevo': 196969, 'Kotlovka': 254237, 'SokolinajaGora': 230000, 'Nagornoe': 261907, 'Basmannoe': 391304, 'Dmitrovskoe': 185526, 'StaroeKrjukovo': 165517, 'TroparevoNikulino': 271428, 'Beskudnikovskoe': 203703, 'BirjulevoZapadnoe': 160294, 'Obruchevskoe': 389221, 'Veshnjaki': 198039, 'Sokolniki': 424672, 'Tverskoe': 443806, 'Solncevo': 207894, 'PoselenieKrasnopahorskoe': 113936, 'ErrorRaionPoselenie': 367686, 'Konkovo': 268595, 'FiliDavydkovo': 326086, 'Levoberezhnoe': 222058, 'ChertanovoCentralnoe': 216016, 'Molzhaninovskoe': 83333, 'Vostochnoe': 150000, 'Marino': 213157, 'Marfino': 243518, 'PoselenieMarushkinskoe': 109090, 'PoselenieVoskresenskoe': 151390, 'Silino': 179166, 'Zamoskvoreche': 335843, 'Dorogomilovo': 435483, 'Perovo': 290000, 'NovoPeredelkino': 164705, 'Alekseevskoe': 320787, 'PoselenieRjazanovskoe': 138597, 'Danilovskoe': 432285, 'Shhukino': 340677, 'JuzhnoeTushino': 269230, 'Losinoostrovskoe': 202631, 'PoselenieRogovskoe': 176315, 'PoselenieMoskovskij': 210798, 'Matushkino': 160000}
    # count = {'VostochnoeIzmajlovo': 154, 'Bibirevo': 230, 'Ramenki': 241, 'TroparevoNikulino': 125, 'Alekseevskoe': 99, 'Obruchevskoe': 182, 'Molzhaninovskoe': 3, 'Rjazanskij': 181, 'ChertanovoJuzhnoe': 272, 'ChertanovoSevernoe': 228, 'Marfino': 85, 'Ajeroport': 123, 'Ljublino': 296, 'Begovoe': 60, 'Altufevskoe': 68, 'Veshnjaki': 212, 'PoselenieShhapovskoe': 2, 'ZapadnoeDegunino': 409, 'Tekstilshhiki': 298, 'Lefortovo': 119, 'Akademicheskoe': 214, 'Krjukovo': 522, 'Sokolniki': 60, 'JuzhnoeTushino': 175, 'Jaroslavskoe': 121, 'PoselenieMoskovskij': 966, 'Kurkino': 61, 'PoselenieRogovskoe': 34, 'Ivanovskoe': 196, 'BirjulevoVostochnoe': 267, 'Savelki': 102, 'Kotlovka': 147, 'Dorogomilovo': 56, 'PoselenieRjazanovskoe': 34, 'Brateevo': 182, 'Jasenevo': 237, 'Zamoskvoreche': 50, 'PoselenieNovofedorovskoe': 156, 'Losinoostrovskoe': 177, 'BirjulevoZapadnoe': 115, 'Timirjazevskoe': 154, 'PoselenieKlenovskoe': 1, 'Savelovskoe': 85, 'Babushkinskoe': 123, 'SevernoeTushino': 281, 'OchakovoMatveevskoe': 255, 'Zjablikovo': 127, 'Gagarinskoe': 78, 'Severnoe': 37, 'OrehovoBorisovoSevernoe': 206, 'TeplyjStan': 164, 'Presnenskoe': 189, 'Pechatniki': 192, 'MarinaRoshha': 116, 'SevernoeButovo': 181, 'Levoberezhnoe': 134, 'Sviblovo': 131, 'SevernoeMedvedkovo': 167, 'PoselenieSosenskoe': 1826, 'Shhukino': 155, 'Sokol': 74, 'Preobrazhenskoe': 151, 'Kapotnja': 49, 'NagatinoSadovniki': 158, 'Mitino': 677, 'Krasnoselskoe': 37, 'Vojkovskoe': 131, 'Meshhanskoe': 94, 'PoseleniePervomajskoe': 142, 'PoselenieShherbinka': 474, 'Horoshevskoe': 134, 'PoselenieMihajlovoJarcevskoe': 1, 'PoselenieVoskresenskoe': 738, 'MoskvorecheSaburovo': 99, 'VostochnoeDegunino': 118, 'Otradnoe': 358, 'Koptevo': 206, 'VyhinoZhulebino': 262, 'Konkovo': 219, 'Tverskoe': 75, 'PoselenieKokoshkino': 19, 'PoselenieMarushkinskoe': 6, 'Golovinskoe': 224, 'Bogorodskoe': 304, 'Matushkino': 111, 'HoroshevoMnevniki': 260, 'Caricyno': 219, 'Goljanovo': 293, 'StaroeKrjukovo': 91, 'PokrovskoeStreshnevo': 162, 'Nizhegorodskoe': 77, 'Hovrino': 178, 'Solncevo': 419, 'Arbat': 15, 'Nekrasovka': 1718, 'PoselenieKievskij': 2, 'Lianozovo': 125, 'Krylatskoe': 102, 'KosinoUhtomskoe': 237, 'Juzhnoportovoe': 126, 'Novokosino': 138, 'PoselenieFilimonkovskoe': 551, 'Donskoe': 135, 'ProspektVernadskogo': 100, 'Cheremushki': 158, 'PoselenieKrasnopahorskoe': 30, 'Lomonosovskoe': 147, 'Silino': 100, 'Zjuzino': 260, 'ChertanovoCentralnoe': 195, 'FilevskijPark': 146, 'Basmannoe': 99, 'SevernoeIzmajlovo': 163, 'Metrogorodok': 58, 'JuzhnoeButovo': 451, 'Kuncevo': 186, 'Izmajlovo': 300, 'Marino': 505, 'NagatinskijZaton': 325, 'Perovo': 246, 'Butyrskoe': 101, 'Kuzminki': 221, 'PoselenieMosrentgen': 19, 'Troickijokrug': 159, 'NovoPeredelkino': 201, 'Mozhajskoe': 197, 'Beskudnikovskoe': 166, 'OrehovoBorisovoJuzhnoe': 208, 'Dmitrovskoe': 174, 'SokolinajaGora': 188, 'Danilovskoe': 199, 'Taganskoe': 173, 'Hamovniki': 90, 'Jakimanka': 81, 'PoselenieDesjonovskoe': 400, 'Novogireevo': 200, 'FiliDavydkovo': 136, 'Rostokino': 66, 'Nagornoe': 302, 'PoselenieVoronovskoe': 7, 'Vnukovo': 43, 'Ostankinskoe': 79, 'Strogino': 301, 'Vostochnoe': 7, 'JuzhnoeMedvedkovo': 143, 'ErrorRaionPoselenie': 52, 'PoselenieVnukovskoe': 1412}

    # min_price = {'Preobrazhenskoe': 765, 'Kuncevo': 823, 'ZapadnoeDegunino': 1056, 'PoselenieKokoshkino': 1291, 'Nizhegorodskoe': 1279, 'Ivanovskoe': 744, 'Obruchevskoe': 787, 'JuzhnoeButovo': 740, 'MoskvorecheSaburovo': 1431, 'Kuzminki': 816, 'Vnukovo': 859, 'BirjulevoZapadnoe': 835, 'PoselenieShherbinka': 722, 'Strogino': 767, 'Alekseevskoe': 903, 'Krasnoselskoe': 1340, 'SevernoeButovo': 1034, 'Lefortovo': 716, 'Danilovskoe': 672, 'Ajeroport': 1230, 'Marfino': 1076, 'NovoPeredelkino': 1129, 'Kurkino': 1325, 'Hovrino': 826, 'Rjazanskij': 1023, 'ProspektVernadskogo': 733, 'ChertanovoJuzhnoe': 925, 'PoselenieKievskij': 1727, 'Vostochnoe': 1997, 'Dmitrovskoe': 1117, 'Tekstilshhiki': 1007, 'TeplyjStan': 726, 'Meshhanskoe': 1826, 'PoselenieMoskovskij': 801, 'Altufevskoe': 921, 'JuzhnoeTushino': 1279, 'Severnoe': 711, 'Hamovniki': 993, 'Akademicheskoe': 929, 'OrehovoBorisovoSevernoe': 1301, 'SokolinajaGora': 1164, 'Lomonosovskoe': 900, 'Sokol': 1004, 'Losinoostrovskoe': 1180, 'Butyrskoe': 1175, 'Mitino': 1158, 'OchakovoMatveevskoe': 789, 'Golovinskoe': 805, 'Izmajlovo': 649, 'TroparevoNikulino': 1208, 'Veshnjaki': 828, 'SevernoeTushino': 1023, 'Nagornoe': 1032, 'SevernoeIzmajlovo': 931, 'Basmannoe': 1346, 'Kapotnja': 2288, 'Jasenevo': 1037, 'PoselenieMosrentgen': 926, 'Donskoe': 865, 'Matushkino': 753, 'PoselenieDesjonovskoe': 963, 'Metrogorodok': 1160, 'PoselenieVoskresenskoe': 1078, 'Shhukino': 1313, 'Presnenskoe': 864, 'PoselenieSosenskoe': 766, 'Timirjazevskoe': 829, 'PoselenieKrasnopahorskoe': 1616, 'FiliDavydkovo': 1375, 'PoselenieVoronovskoe': 996, 'Zjablikovo': 784, 'Tverskoe': 1132, 'Bibirevo': 1124, 'Juzhnoportovoe': 1594, 'Zamoskvoreche': 1539, 'Rostokino': 1409, 'PoselenieMarushkinskoe': 1988, 'VyhinoZhulebino': 849, 'Sviblovo': 1116, 'NagatinskijZaton': 1248, 'Sokolniki': 1080, 'Konkovo': 887, 'Bogorodskoe': 830, 'PoselenieVnukovskoe': 859, 'NagatinoSadovniki': 1408, 'PoselenieMihajlovoJarcevskoe': 2098, 'MarinaRoshha': 959, 'PoselenieRogovskoe': 711, 'Molzhaninovskoe': 1374, 'Brateevo': 1141, 'Novogireevo': 1125, 'StaroeKrjukovo': 975, 'Krjukovo': 779, 'Beskudnikovskoe': 945, 'PoseleniePervomajskoe': 985, 'Dorogomilovo': 1665, 'Koptevo': 895, 'Troickijokrug': 1070, 'Gagarinskoe': 2236, 'PoselenieShhapovskoe': 1194, 'Cheremushki': 1148, 'Horoshevskoe': 1303, 'ChertanovoSevernoe': 774, 'Ljublino': 758, 'Lianozovo': 1342, 'Jakimanka': 786, 'Vojkovskoe': 763, 'Savelovskoe': 1753, 'Perovo': 665, 'PoselenieFilimonkovskoe': 1123, 'Ostankinskoe': 1162, 'Levoberezhnoe': 713, 'Nekrasovka': 836, 'Pechatniki': 804, 'VostochnoeDegunino': 1118, 'Begovoe': 1514, 'Otradnoe': 626, 'ChertanovoCentralnoe': 852, 'Novokosino': 822, 'Silino': 882, 'Caricyno': 969, 'Kotlovka': 1010, 'Krylatskoe': 1462, 'Mozhajskoe': 812, 'BirjulevoVostochnoe': 754, 'Arbat': 2010, 'PokrovskoeStreshnevo': 853, 'Solncevo': 1247, 'Savelki': 879, 'Zjuzino': 894, 'JuzhnoeMedvedkovo': 770, 'PoselenieNovofedorovskoe': 735, 'VostochnoeIzmajlovo': 781, 'Marino': 893, 'Jaroslavskoe': 977, 'KosinoUhtomskoe': 731, 'PoselenieRjazanovskoe': 927, 'ErrorRaionPoselenie': 883, 'Babushkinskoe': 1162, 'Goljanovo': 899, 'OrehovoBorisovoJuzhnoe': 1192, 'Ramenki': 1296, 'Taganskoe': 828, 'FilevskijPark': 837, 'SevernoeMedvedkovo': 1186, 'HoroshevoMnevniki': 746}
    # max_price = {'ChertanovoSevernoe': 7403, 'OrehovoBorisovoSevernoe': 5956, 'Solncevo': 6415, 'Beskudnikovskoe': 6322, 'Hamovniki': 14935, 'Golovinskoe': 6670, 'FilevskijPark': 7727, 'Lomonosovskoe': 8946, 'PoselenieSosenskoe': 11858, 'PoselenieKrasnopahorskoe': 3149, 'Izmajlovo': 7839, 'SevernoeIzmajlovo': 7216, 'Tekstilshhiki': 10096, 'MarinaRoshha': 15495, 'Mitino': 7069, 'PoselenieFilimonkovskoe': 3816, 'Rjazanskij': 6517, 'Donskoe': 13434, 'PoselenieMosrentgen': 4098, 'PoselenieKievskij': 3576, 'Bibirevo': 5794, 'Vojkovskoe': 7097, 'TroparevoNikulino': 8056, 'Ramenki': 15518, 'Bogorodskoe': 10831, 'Cheremushki': 8980, 'Begovoe': 8493, 'Caricyno': 7196, 'Lefortovo': 8632, 'PoselenieMoskovskij': 6746, 'Marino': 6487, 'SokolinajaGora': 7184, 'Zjuzino': 8843, 'Molzhaninovskoe': 1374, 'Jakimanka': 12728, 'Jaroslavskoe': 5290, 'Brateevo': 5637, 'Krasnoselskoe': 9572, 'PoselenieNovofedorovskoe': 3365, 'Tverskoe': 13687, 'Shhukino': 10875, 'Sokol': 8442, 'JuzhnoeTushino': 8905, 'Rostokino': 7269, 'Novokosino': 5814, 'Horoshevskoe': 9809, 'Troickijokrug': 4501, 'Ivanovskoe': 5583, 'TeplyjStan': 7784, 'PoselenieDesjonovskoe': 3211, 'VyhinoZhulebino': 5785, 'Silino': 6079, 'Marfino': 6507, 'JuzhnoeButovo': 5944, 'Kapotnja': 4943, 'Pechatniki': 5874, 'PoselenieMarushkinskoe': 2654, 'OrehovoBorisovoJuzhnoe': 6025, 'Altufevskoe': 7461, 'PoselenieMihajlovoJarcevskoe': 2098, 'Savelki': 5126, 'NagatinoSadovniki': 7811, 'PokrovskoeStreshnevo': 12441, 'Veshnjaki': 5584, 'PoselenieVoskresenskoe': 3930, 'Presnenskoe': 11545, 'Kuncevo': 7927, 'ZapadnoeDegunino': 5678, 'Konkovo': 7554, 'Losinoostrovskoe': 5830, 'Ostankinskoe': 8251, 'Metrogorodok': 5926, 'Kuzminki': 6688, 'Mozhajskoe': 7996, 'NagatinskijZaton': 6669, 'JuzhnoeMedvedkovo': 5969, 'SevernoeMedvedkovo': 6575, 'FiliDavydkovo': 10064, 'Timirjazevskoe': 7644, 'Strogino': 11802, 'Kurkino': 6982, 'Novogireevo': 6642, 'Vostochnoe': 4631, 'Ajeroport': 8703, 'Ljublino': 9743, 'PoselenieShhapovskoe': 1489, 'Krylatskoe': 10345, 'NovoPeredelkino': 5025, 'SevernoeTushino': 6775, 'ChertanovoCentralnoe': 13802, 'Krjukovo': 4545, 'PoselenieVnukovskoe': 4362, 'Alekseevskoe': 9201, 'ProspektVernadskogo': 16650, 'Savelovskoe': 7716, 'VostochnoeDegunino': 5560, 'Meshhanskoe': 10023, 'OchakovoMatveevskoe': 7422, 'Perovo': 6631, 'Nagornoe': 7141, 'Zamoskvoreche': 9993, 'Butyrskoe': 7485, 'StaroeKrjukovo': 5069, 'Danilovskoe': 7987, 'Goljanovo': 7783, 'Dmitrovskoe': 5467, 'Taganskoe': 11847, 'Levoberezhnoe': 7149, 'Akademicheskoe': 9213, 'Juzhnoportovoe': 9268, 'Koptevo': 9571, 'BirjulevoVostochnoe': 6195, 'Severnoe': 4886, 'BirjulevoZapadnoe': 4798, 'Babushkinskoe': 10307, 'Lianozovo': 6143, 'VostochnoeIzmajlovo': 9657, 'HoroshevoMnevniki': 10024, 'PoselenieRogovskoe': 3735, 'ErrorRaionPoselenie': 11236, 'PoselenieRjazanovskoe': 3950, 'Matushkino': 4852, 'PoselenieShherbinka': 4295, 'Otradnoe': 6931, 'Nizhegorodskoe': 5857, 'ChertanovoJuzhnoe': 6215, 'Arbat': 12667, 'Preobrazhenskoe': 6943, 'Nekrasovka': 6450, 'Sviblovo': 7031, 'Kotlovka': 7734, 'PoseleniePervomajskoe': 3252, 'PoselenieVoronovskoe': 2751, 'Zjablikovo': 5832, 'Sokolniki': 12001, 'KosinoUhtomskoe': 5162, 'Jasenevo': 6653, 'PoselenieKokoshkino': 3350, 'MoskvorecheSaburovo': 7831, 'Vnukovo': 4364, 'SevernoeButovo': 6496, 'Dorogomilovo': 11039, 'Basmannoe': 10817, 'Hovrino': 7127, 'Obruchevskoe': 11881, 'Gagarinskoe': 9849}
    # count = {'FiliDavydkovo': 123, 'Strogino': 281, 'Pechatniki': 175, 'Taganskoe': 149, 'Losinoostrovskoe': 164, 'Savelki': 86, 'Krylatskoe': 99, 'PoselenieVoskresenskoe': 707, 'BirjulevoVostochnoe': 243, 'Kapotnja': 46, 'ProspektVernadskogo': 96, 'Kuncevo': 167, 'Konkovo': 209, 'Mitino': 663, 'Hovrino': 162, 'Basmannoe': 94, 'ChertanovoCentralnoe': 181, 'Sviblovo': 123, 'Levoberezhnoe': 124, 'PoseleniePervomajskoe': 132, 'PoselenieKrasnopahorskoe': 22, 'Hamovniki': 90, 'Mozhajskoe': 179, 'SevernoeButovo': 169, 'PoselenieMoskovskij': 923, 'Dmitrovskoe': 161, 'Sokol': 63, 'OrehovoBorisovoJuzhnoe': 198, 'Begovoe': 59, 'Juzhnoportovoe': 121, 'Babushkinskoe': 112, 'PoselenieVnukovskoe': 1369, 'TroparevoNikulino': 116, 'Molzhaninovskoe': 1, 'PoselenieFilimonkovskoe': 495, 'Ajeroport': 119, 'Lomonosovskoe': 141, 'Kuzminki': 208, 'NagatinoSadovniki': 147, 'VostochnoeDegunino': 108, 'JuzhnoeButovo': 413, 'Perovo': 230, 'SevernoeTushino': 274, 'PoselenieDesjonovskoe': 358, 'Lianozovo': 110, 'Ljublino': 271, 'Jaroslavskoe': 114, 'Lefortovo': 110, 'ZapadnoeDegunino': 394, 'Gagarinskoe': 75, 'Metrogorodok': 55, 'Caricyno': 210, 'ErrorRaionPoselenie': 595, 'Presnenskoe': 168, 'Preobrazhenskoe': 141, 'Matushkino': 90, 'Akademicheskoe': 189, 'Alekseevskoe': 94, 'HoroshevoMnevniki': 231, 'Dorogomilovo': 53, 'Tverskoe': 73, 'Altufevskoe': 57, 'Obruchevskoe': 170, 'Novokosino': 129, 'Koptevo': 192, 'Sokolniki': 59, 'Savelovskoe': 78, 'Rostokino': 56, 'Meshhanskoe': 88, 'PoselenieShherbinka': 438, 'Izmajlovo': 288, 'Kotlovka': 138, 'MarinaRoshha': 113, 'Tekstilshhiki': 288, 'Bibirevo': 209, 'TeplyjStan': 154, 'Donskoe': 126, 'Kurkino': 56, 'Silino': 86, 'Rjazanskij': 169, 'Jakimanka': 73, 'Bogorodskoe': 279, 'Golovinskoe': 204, 'VyhinoZhulebino': 241, 'Krasnoselskoe': 36, 'ChertanovoJuzhnoe': 255, 'PoselenieSosenskoe': 1773, 'FilevskijPark': 135, 'ChertanovoSevernoe': 194, 'Goljanovo': 261, 'MoskvorecheSaburovo': 91, 'Nagornoe': 289, 'Vojkovskoe': 125, 'BirjulevoZapadnoe': 108, 'Butyrskoe': 90, 'Zamoskvoreche': 49, 'Vostochnoe': 7, 'Solncevo': 399, 'Beskudnikovskoe': 147, 'PokrovskoeStreshnevo': 159, 'Jasenevo': 222, 'Cheremushki': 150, 'Krjukovo': 481, 'KosinoUhtomskoe': 212, 'Ivanovskoe': 177, 'Troickijokrug': 148, 'Marfino': 78, 'Brateevo': 167, 'VostochnoeIzmajlovo': 145, 'NovoPeredelkino': 139, 'Zjuzino': 240, 'Arbat': 15, 'PoselenieMosrentgen': 17, 'PoselenieKievskij': 2, 'PoselenieNovofedorovskoe': 147, 'JuzhnoeMedvedkovo': 119, 'PoselenieRjazanovskoe': 31, 'OchakovoMatveevskoe': 242, 'JuzhnoeTushino': 162, 'PoselenieKokoshkino': 16, 'PoselenieMihajlovoJarcevskoe': 1, 'PoselenieMarushkinskoe': 4, 'Novogireevo': 187, 'SokolinajaGora': 172, 'PoselenieRogovskoe': 30, 'SevernoeMedvedkovo': 149, 'PoselenieVoronovskoe': 7, 'Shhukino': 146, 'StaroeKrjukovo': 72, 'Nizhegorodskoe': 71, 'Ostankinskoe': 75, 'Marino': 467, 'Horoshevskoe': 130, 'Ramenki': 230, 'Nekrasovka': 1574, 'Timirjazevskoe': 139, 'Otradnoe': 328, 'SevernoeIzmajlovo': 154, 'OrehovoBorisovoSevernoe': 193, 'Vnukovo': 40, 'Danilovskoe': 189, 'Severnoe': 33, 'PoselenieShhapovskoe': 2, 'Veshnjaki': 198, 'Zjablikovo': 116, 'NagatinskijZaton': 319}
    # avg_new = {'PoselenieFilimonkovskoe': 1965, 'Donskoe': 5585, 'Babushkinskoe': 4686, 'BirjulevoVostochnoe': 3438, 'Nekrasovka': 2913, 'Cheremushki': 5178, 'Lomonosovskoe': 5627, 'Ljublino': 4066, 'Preobrazhenskoe': 4550, 'Arbat': 7770, 'Ramenki': 5171, 'Shhukino': 5078, 'KosinoUhtomskoe': 3730, 'Perovo': 3953, 'Zjuzino': 4696, 'PoselenieNovofedorovskoe': 1247, 'Pechatniki': 3980, 'Tekstilshhiki': 3915, 'PoselenieVnukovskoe': 2734, 'Horoshevskoe': 5037, 'TroparevoNikulino': 4854, 'PoselenieVoronovskoe': 2024, 'JuzhnoeButovo': 3714, 'Kapotnja': 3514, 'Kurkino': 4276, 'Mitino': 4274, 'PoselenieMoskovskij': 2513, 'Jaroslavskoe': 3694, 'MoskvorecheSaburovo': 4469, 'VyhinoZhulebino': 3887, 'PoselenieShhapovskoe': 1341, 'PokrovskoeStreshnevo': 4612, 'PoselenieRjazanovskoe': 2315, 'Akademicheskoe': 5114, 'Troickijokrug': 2336, 'Bibirevo': 4116, 'Beskudnikovskoe': 4041, 'NovoPeredelkino': 3566, 'Sokolniki': 5777, 'ChertanovoCentralnoe': 4463, 'Izmajlovo': 4282, 'PoselenieMarushkinskoe': 2377, 'Jasenevo': 4375, 'Matushkino': 3350, 'PoselenieKievskij': 2651, 'Marino': 4152, 'ChertanovoSevernoe': 4320, 'SevernoeIzmajlovo': 4067, 'PoselenieKokoshkino': 2214, 'Lianozovo': 4330, 'NagatinskijZaton': 4162, 'PoseleniePervomajskoe': 1735, 'SevernoeTushino': 4647, 'TeplyjStan': 4241, 'Vojkovskoe': 4502, 'Rostokino': 4480, 'PoselenieMosrentgen': 2958, 'Nagornoe': 4623, 'HoroshevoMnevniki': 4827, 'PoselenieDesjonovskoe': 2349, 'Savelovskoe': 4886, 'Golovinskoe': 4122, 'JuzhnoeTushino': 4316, 'Brateevo': 3883, 'Zjablikovo': 4203, 'Severnoe': 3340, 'Meshhanskoe': 6006, 'Novokosino': 3912, 'FilevskijPark': 4501, 'Kuzminki': 4079, 'Rjazanskij': 3999, 'Tverskoe': 6739, 'Krylatskoe': 5385, 'PoselenieVoskresenskoe': 2856, 'Caricyno': 4174, 'PoselenieShherbinka': 2215, 'Silino': 3351, 'ProspektVernadskogo': 5542, 'Hamovniki': 7796, 'VostochnoeIzmajlovo': 3917, 'Krjukovo': 3013, 'Begovoe': 5377, 'Strogino': 5068, 'Konkovo': 4830, 'Alekseevskoe': 4971, 'ChertanovoJuzhnoe': 4248, 'Basmannoe': 5608, 'Sokol': 4826, 'Metrogorodok': 4018, 'Hovrino': 4233, 'Vostochnoe': 3512, 'OrehovoBorisovoJuzhnoe': 4112, 'Gagarinskoe': 6011, 'ErrorRaionPoselenie': 2634, 'NagatinoSadovniki': 4447, 'Ivanovskoe': 3801, 'BirjulevoZapadnoe': 3509, 'Butyrskoe': 4517, 'Novogireevo': 4292, 'Veshnjaki': 3730, 'PoselenieRogovskoe': 1255, 'ZapadnoeDegunino': 2898, 'SevernoeButovo': 4257, 'Losinoostrovskoe': 4216, 'FiliDavydkovo': 4771, 'Jakimanka': 5423, 'Zamoskvoreche': 6600, 'Bogorodskoe': 4141, 'Goljanovo': 3972, 'SevernoeMedvedkovo': 4276, 'Molzhaninovskoe': 1374, 'OrehovoBorisovoSevernoe': 3802, 'Savelki': 3375, 'Altufevskoe': 4068, 'Dmitrovskoe': 3868, 'Vnukovo': 3313, 'Solncevo': 3176, 'Juzhnoportovoe': 4826, 'Krasnoselskoe': 5860, 'Kuncevo': 4682, 'Presnenskoe': 5770, 'Nizhegorodskoe': 4010, 'Marfino': 4216, 'Taganskoe': 5522, 'StaroeKrjukovo': 3286, 'Koptevo': 4409, 'Kotlovka': 4659, 'JuzhnoeMedvedkovo': 4240, 'Timirjazevskoe': 4581, 'MarinaRoshha': 4698, 'Ajeroport': 5041, 'PoselenieMihajlovoJarcevskoe': 2098, 'OchakovoMatveevskoe': 4228, 'Ostankinskoe': 5074, 'Lefortovo': 4630, 'Danilovskoe': 4758, 'PoselenieSosenskoe': 2590, 'VostochnoeDegunino': 4024, 'Levoberezhnoe': 3993, 'Sviblovo': 4310, 'Dorogomilovo': 6440, 'Otradnoe': 4205, 'Obruchevskoe': 4995, 'PoselenieKrasnopahorskoe': 1962, 'Mozhajskoe': 4151, 'SokolinajaGora': 4452}


def filter_kremlin_km(raw_data, is_production=None, is_convert=1):
    # TODO check 3 types of distances: in city, out of, and out of city ring
    # Check range of rlimit
    # rlimit = np.percentile(raw_data['kremlin_km'].values, 98.0)
    rlimit = np.percentile(raw_data['kremlin_km'].values, 99.85)
    # llimit = max(1, np.percentile(raw_data['kremlin_km'].values, 0.5))

    # print(rlimit, llimit)

    # raw_data.loc[raw_data['kremlin_km'] < llimit, 'kremlin_km'] = np.NAN
    # raw_data.loc[raw_data['kremlin_km'] < llimit, 'kremlin_km'] = np.mean(raw_data['kremlin_km'])
    if not is_production:
        raw_data.loc[raw_data['kremlin_km'] > rlimit, 'kremlin_km'] = rlimit

    if is_convert:
        raw_data['kremlin_km'] = np.log(raw_data['kremlin_km'])

    # rlimit = min(rlimit, 30)
    # show_correlations(raw_data, 'kremlin_km')
    # asd()

    # raw_data['fixed_basket'] = 0
    # raw_data.loc[raw_data['kremlin_km'] > rlimit, 'fixed_basket'] = 1

    # ?
    # raw_data.loc[raw_data['kremlin_km'] > rlimit, 'kremlin_km'] = np.NAN
    # set_av(raw_data, 'kremlin_km')

    # 0 - Investment

    # print(raw_data.loc[(raw_data['product_type'] == 0) & (
    #     np.isnan(raw_data['build_year']))])
    # asd()

    # gr_invest = raw_data.loc[raw_data['product_type'] == 0]
    # gr_own = raw_data.loc[raw_data['product_type'] == 1]

    # print(gr_invest.loc[gr_invest['price_doc'] == 1e6][['full_sq', 'life_sq',
    #     'max_floor', 'kremlin_km', 'sub_area', 'build_year', 'sale_year']])
    #  print(gr_invest.loc[gr_invest['price_doc'] == 1e6][['full_sq', 'life_sq',
    #     'max_floor', 'kremlin_km', 'sub_area', 'build_year']])

    # Инвестиционная должна быть близко к метро
    # в чистом зеленом районе

    # print(gr_invest)
    # show_hist(gr_invest['price_doc'])
    # show_hist(gr_own['price_doc'])
    # asd()

    # show_feature_over_time(gr_invest, 'price_doc')
    # show_feature_over_time(gr_own, 'price_doc')
    # show_pair_plot(gr_invest, ['kremlin_km', 'price_doc', 'product_type'], 'product_type')
    # asd()

    # show_pair_plot(gr_invest, ['kremlin_km', 'price_doc', 'product_type'], 'product_type')
    # show_pair_plot(gr_own, ['kremlin_km', 'price_doc', 'product_type'], 'product_type')

    # show_correlations(gr_invest, 'kremlin_km')
    # show_correlations(gr_own, 'kremlin_km')

    # asd()
    # describe(raw_data, 'kremlin_km')

    # 'mkad_km', 'water_1line'
    # raw_data['full_sq'] = raw_data['full_sq'] ** 1.45
    # describe(raw_data, 'full_sq', nlo=30)
    # visualize_feature_over_time()

    # raw_data['center_km'] = np.NAN
    # raw_data.loc[raw_data['kremlin_km'] < 5.5, 'center_km'] = 1
    # raw_data.loc[(raw_data['kremlin_km'] < 5.5) & , 'center_km'] = 1
    # raw_data.loc[(raw_data['kremlin_km'] >= 5.5) & (raw_data['kremlin_km'] < 15), 'center_km'] = 2.2
    # raw_data.loc[(raw_data['mkad_km'] > 5) & (raw_data['kremlin_km'] > 25), 'center_km'] = 3
    # raw_data.loc[(raw_data['mkad_km'] > 10) & (raw_data['kremlin_km'] > 25), 'center_km'] = 3.8
    # raw_data.loc[(raw_data['mkad_km'] > 15) & (raw_data['kremlin_km'] > 25), 'center_km'] = 3
    # raw_data.loc[(raw_data['mkad_km'] > 30) & (raw_data['kremlin_km'] > 25), 'center_km'] = 4.1
    # raw_data['center_km'] = raw_data['center_km'].fillna(np.mean(raw_data['center_km']))
    # raw_data['kremlin_km'] = raw_data['center_km']

    # raw_data['kremlin_km'] = raw_data['kremlin_km'] < 5.5


def filter_product_type(raw_data):
    raw_data['product_type'] = raw_data['product_type'].map({'Investment': 0, 'OwnerOccupier': 1})

    # raw_data.loc[np.isnan(raw_data['product_type']) & np.isnan(raw_data['build_year'])] = 0
    # For test data
    # All ( 5604 : 1 ) with Nan build_year & Nan life_sq are Owners - 1
    raw_data.loc[np.isnan(raw_data['build_year']) &
                 np.isnan(raw_data['life_sq']) &
                 np.isnan(raw_data['product_type'])
                 , 'product_type'] = 1


def filter_build_year(raw_data):
    # TODO
    # print(train.loc[train['build_year'].isnull()])
    # print(len(train[train['sale_year'] < train['build_year']]))

    # raw_data.loc[raw_data['build_year'] < 1000, 'build_year'] = 0

    raw_data.insert(0, 'build_year_error', np.NAN)
    raw_data.loc[raw_data['build_year'] <= 10, 'build_year_error'] = raw_data['build_year']
    raw_data.loc[raw_data['build_year'] < 1800, 'build_year'] = np.NAN
    raw_data.loc[raw_data['build_year'] > 2020, 'build_year'] = np.NAN

    raw_data.insert(0, 'year_old', np.NAN)
    raw_data.loc[~np.isnan(raw_data['build_year']), 'year_old'] = \
        raw_data['sale_year'] - raw_data['build_year']

    # Addition of field worsens model
    # raw_data.insert(0, 'build_year_error', 0)
    # raw_data.loc[raw_data['build_year'] < 1500, 'build_year_error'] = 1

    # Useless
    # raw_data.insert(0, 'is_larger_build_year', 0)
    # raw_data.loc[raw_data['build_year'] > 2020, 'is_larger_build_year'] = 1

    # raw_data.insert(0, 'life_sq_pers', np.NAN)
    # raw_data['build_old'] = (raw_data['full_sq'] - raw_data['life_sq']) / raw_data['full_sq']

    # set_av(raw_data, 'build_year')
    return
    # print(raw_data[np.isnan(raw_data['build_year'])])
    # asd()

    # describe(raw_data, 'build_year')
    # asd()

    # TODO explore nearest buildings
    # raion_build_count_with_builddate_info

    # print(raw_data[['build_year', 'build_count_before_1920',
    #                 'build_count_1921-1945', 'build_count_1946-1970',
    #                 'build_count_1971-1995', 'build_count_after_1995', 'price_doc']])

    raw_data.loc[np.isnan(raw_data['build_count_before_1920']),
                 'build_count_before_1920'] = 0
    raw_data.loc[np.isnan(raw_data['build_count_1921-1945']),
                 'build_count_1921-1945'] = 0
    raw_data.loc[np.isnan(raw_data['build_count_1946-1970']),
                 'build_count_1946-1970'] = 0
    raw_data.loc[np.isnan(raw_data['build_count_1971-1995']),
                 'build_count_1971-1995'] = 0
    raw_data.loc[np.isnan(raw_data['build_count_after_1995']),
                 'build_count_after_1995'] = 0

    t = raw_data[~np.isnan(raw_data['raion_build_count_with_builddate_info'])]
    t = t[['build_year', 'build_count_before_1920',
           'build_count_1921-1945', 'build_count_1946-1970',
           'build_count_1971-1995', 'build_count_after_1995',
           'max_floor', 'state', 'sub_area', 'material', 'kremlin_km', 'product_type']]

    # Predict build year if build year neighbor is set
    data_to_predict = t[np.isnan(t['build_year'])]

    simple_predict(t, t, data_to_predict, 'build_year')
    raw_data['build_year'] = t['build_year']
    # raw_data.loc[raw_data['build_year'] < 1800, 'build_year'] = 1800
    # raw_data.loc[raw_data['build_year'] > 2025, 'build_year'] = 2025
    # Set av build year for other
    # set_av(raw_data, 'build_year')

    return

    raw_data.loc[np.isnan(raw_data['build_year']), 'build_year'] = 0

    # show_correlations(raw_data, 'build_year')
    # asd()

    raw_data['build_year'] = raw_data['build_year'].astype(int)

    # raw_data.loc[raw_data['build_year'] < 1920, 'build_year'] = 1
    # raw_data.loc[(raw_data['build_year'] >= 1920)
    #              & (raw_data['build_year'] < 1945), 'build_year'] = 2
    # raw_data.loc[(raw_data['build_year'] >= 1945)
    #              & (raw_data['build_year'] < 1970), 'build_year'] = 3
    # raw_data.loc[(raw_data['build_year'] >= 1970)
    #              & (raw_data['build_year'] < 1995), 'build_year'] = 4
    # raw_data.loc[raw_data['build_year'] >= 1995, 'build_year'] = 5

    # 'build_count_before_1920',
    # 'build_count_1921-1945', 'build_count_1946-1970',
    # 'build_count_1971-1995', 'build_count_after_1995',

    # show_correlations(raw_data, 'build_year')
    # show_pair_plot(raw_data, ['build_year', 'price_doc'])

    # describe(raw_data, 'build_year')
    # asd()

    # print(test['build_year'])
    # print(len(predict_year))
    #
    # plt.hist(predict_year, bins=10)
    # plt.show()

    # default_validate(raw_data, 'build_year', 1800, 2050, custom_default=2011)

    # raw_data['build_year'] = raw_data['build_year'].astype(int)

    # describe(raw_data, 'build_year')
    # asd()
    # print(len(train[train['sale_year'] < train['build_year']]))
    # print(len(train[train['build_year'] > train['build_year']]))
    # print(stats.pearsonr(train[['sale_year']], train[['price_doc']]))
    # sns.set_style("whitegrid")
    # ax = sns.barplot(x="sale_month", y="price_doc", data=train)
    # plt.show()
    # s()

    # sns.set_style("whitegrid")
    # sns.barplot(x="build_year", y="price_doc", data=train)
    # plt.show()
    # sys.exit(0)
    # t = raw_data[~np.isnan(train['build_year']) & train['state'] > 0]
    # sns.pairplot(t[['build_year', 'state']], kind="reg")
    # plt.show()
    # print(raw_data[np.isnan(raw_data['material'])])
    # print(len(raw_data[np.isnan(raw_data['state']) & ~np.isnan(raw_data['build_year'])]))
    # print(len(raw_data))

    # t = raw_data[~np.isnan(train['build_year']) & train['state'] > 0]
    # sns.pairplot(t[['build_year', 'state']], kind="reg")
    # plt.show()


def filter_ecology(data):

    # pr(raw_data[['ecology']])
    # print(raw_data.groupby(['ecology'])['ecology'].agg(['count']))

    data.loc[data['ecology'] == 'no data', 'ecology'] = 0
    data['ecology'] = data['ecology']\
        .map({'excellent': 5, 'good': 4, 'satisfactory': 3, 'poor': 2})

    # set_av(data, 'ecology')

    return

    show_correlations(data, 'ecology', 'price_doc')

    # plt.hist(data[~np.isnan(data['ecology'])]['ecology'], bins=10)
    # plt.show()
    # asd()

    # set_av(data, 'ecology')
    # data['ecology'] = np.log(data['ecology'])
    # show_correlations(data, 'ecology', 'price_doc')
    # describe(raw_data, 'ecology')

    data['radiation_raion'] = data['radiation_raion'].map({'yes': 1, 'no': 0})

    t = data[['ecology', 'green_part_5000', 'oil_chemistry_km',
              'big_road1_km', 'power_transmission_line_km', 'thermal_power_plant_km',
              # 'radiation_km', 'radiation_raion',
              ]]

    print(t)
    asd()

    data_to_predict = t[np.isnan(t['ecology'])]
    # simple_predict(data, t, data_to_predict, 'ecology', 1, 1, 1)
    simple_predict(data, t, data_to_predict, 'ecology')

    show_correlations(data, 'ecology', 'price_doc')
    asd()

    # show_correlations(data, 'ecology', 'price_doc')
    # show_pair_plot(data, ['ecology', 'price_doc'])
    # data['ecology'] **= -2
    # describe(data, 'ecology')


def simple_predict(data, train, data_to_predict, predict_field_name, show_features=None,
                   show_predictions=None, show_predictions_hist=None):
    if len(data_to_predict) > 0:
        predict_columns = list(train)[1:]
        train = train[~np.isnan(train[predict_field_name])]
        train = train.values
        X_train = train[0::, 1::]
        y_train = train[0::, :1:]

        regr = GradientBoostingRegressor(max_depth=7)
        regr.fit(X_train, y_train)
        _predict = regr.predict(data_to_predict.values[0::, 1::])
        if show_features:
            print(predict_columns)
            print(regr.feature_importances_)

        if show_predictions:
            print(_predict)
            print('Mean:', np.mean(_predict))

        if show_predictions_hist:
            plt.hist(_predict, bins=10)
            plt.show()

        data.loc[np.isnan(data[predict_field_name]), predict_field_name]\
            = np.round(_predict)


# def filter_kinder_garden(raw_data):
#     raw_data['kindergarten_km'] *= 100
#     raw_data['kindergarten_km'] = raw_data['kindergarten_km'].astype('int')
#     rlimit = np.percentile(train['kindergarten_km'].values, 99.5)
#     llimit = np.percentile(train['kindergarten_km'].values, 0.5)
#     train.loc[raw_data['kindergarten_km'] > rlimit, 'kindergarten_km'] = np.NAN
#     set_av(raw_data, 'kindergarten_km')


def format_default(dfa):
    dfa["fullzero"] = (dfa.full_sq == 0)
    dfa["fulltiny"] = (dfa.full_sq < 4)
    dfa["fullhuge"] = (dfa.full_sq > 2000)
    dfa["lnfull"] = np.log(dfa.full_sq + 1)

    dfa["nolife"] = dfa.life_sq.isnull()
    dfa.life_sq = dfa.life_sq.fillna(dfa.life_sq.median())
    dfa["lifezero"] = (dfa.life_sq == 0)
    dfa["lifetiny"] = (dfa.life_sq < 4)
    dfa["lifehuge"] = (dfa.life_sq > 2000)
    dfa["lnlife"] = np.log(dfa.life_sq + 1)

    dfa["nofloor"] = dfa.floor.isnull()
    dfa.floor = dfa.floor.fillna(dfa.floor.median())
    dfa["floor1"] = (dfa.floor == 1)
    dfa["floor0"] = (dfa.floor == 0)
    dfa["floorhuge"] = (dfa.floor > 50)
    dfa["lnfloor"] = np.log(dfa.floor + 1)

    dfa["nomax"] = dfa.max_floor.isnull()
    dfa.max_floor = dfa.max_floor.fillna(dfa.max_floor.median())
    dfa["max1"] = (dfa.max_floor == 1)
    dfa["max0"] = (dfa.max_floor == 0)
    dfa["maxhuge"] = (dfa.max_floor > 80)
    dfa["lnmax"] = np.log(dfa.max_floor + 1)

    dfa["norooms"] = dfa.num_room.isnull()
    dfa.num_room = dfa.num_room.fillna(dfa.num_room.median())
    dfa["zerorooms"] = (dfa.num_room == 0)
    dfa["lnrooms"] = np.log(dfa.num_room + 1)

    dfa["nokitch"] = dfa.kitch_sq.isnull()
    dfa.kitch_sq = dfa.kitch_sq.fillna(dfa.kitch_sq.median())
    dfa["kitch1"] = (dfa.kitch_sq == 1)
    dfa["kitch0"] = (dfa.kitch_sq == 0)
    dfa["kitchhuge"] = (dfa.kitch_sq > 400)
    dfa["lnkitch"] = np.log(dfa.kitch_sq + 1)

    dfa["material0"] = dfa.material.isnull()
    dfa["material1"] = (dfa.material==1)
    dfa["material2"] = (dfa.material==2)
    dfa["material3"] = (dfa.material==3)
    dfa["material4"] = (dfa.material==4)
    dfa["material5"] = (dfa.material==5)
    dfa["material6"] = (dfa.material==6)

    # "state" isn't explained but it looks like an ordinal number, so for now keep numeric
    dfa.loc[dfa.state > 5,"state"] = np.NaN  # Value 33 seems to be invalid; others all 1-4
    dfa.state = dfa.state.fillna(dfa.state.median())

    # product_type gonna be ugly because there are missing values in the test set but not training
    # Check for the same problem with other variables
    dfa["owner_occ"] = (dfa.product_type=='OwnerOccupier')
    dfa.owner_occ.fillna(dfa.owner_occ.mean())

    dfa = pd.get_dummies(dfa, columns=['sub_area'], drop_first=True)

    # Build year is ugly
    # Can be missing
    # Can be zero
    # Can be one
    # Can be some ridiculous pre-Medieval number
    # Can be some invalid huge number like 20052009
    # Can be some other invalid huge number like 4965
    # Can be a reasonable number but later than purchase year
    # Can be equal to purchase year
    # Can be a reasonable nubmer before purchase year

    dfa.loc[dfa.build_year>2030,"build_year"] = np.NaN
    dfa["nobuild"] = dfa.build_year.isnull()
    dfa["sincebuild"] = pd.to_datetime(dfa.timestamp).dt.year - dfa.build_year
    dfa.sincebuild.fillna(dfa.sincebuild.median(),inplace=True)
    dfa["futurebuild"] = (dfa.sincebuild < 0)
    dfa["newhouse"] = (dfa.sincebuild==0)
    dfa["tooold"] = (dfa.sincebuild>1000)
    dfa["build0"] = (dfa.build_year==0)
    dfa["build1"] = (dfa.build_year==1)
    dfa["untilbuild"] = -dfa.sincebuild.apply(np.min, args=[0]) # How many years until planned build
    dfa["lnsince"] = dfa.sincebuild.mul(dfa.sincebuild>0).add(1).apply(np.log)


# +1. Check old before post
# +2. Check local cv with fix
# +3. Check with error_marker
# 4. Check again with error_marker
def fix_data_error(df_train, df_test, fx):
    # print(fx.index.tolist()) print(fx.index.values)
    fix_indexes = fx.index.values

    # print(df_train.loc[314])
    # asd()
    # show_values_stat(df_test, 'error_rows')

    df_train.update(fx, overwrite=True)
    df_test.update(fx, overwrite=True)

    # print(df_train.shape)
    # print(df_test.shape)

    # bad_train_index = [i for i in fix_indexes if i <= df_train.index.max()]
    # bad_test_index = [i for i in fix_indexes if i > df_train.index.max()]

    # df_train.insert(0, 'error_rows', 0)
    # df_test.insert(0, 'error_rows', 0)

    # mark updated rows with extra field
    # df_train.loc[bad_train_index, 'error_rows'] = 1
    # df_test.loc[bad_test_index, 'error_rows'] = 1

    # print(df_train.loc[313])
    # print(df_train.loc[314])
    # asd()


def manual_fix_kremlin_error(df_train, df_test):

    for df in [df_train, df_test]:
        for k in ['area_m', 'raion_popul', 'green_zone_part', 'indust_part',
                  'children_preschool', 'preschool_quota',
                  'preschool_education_centers_raion', 'children_school',
                  'school_quota', 'school_education_centers_raion',
                  'school_education_centers_top_20_raion', 'hospital_beds_raion',
                  'healthcare_centers_raion', 'university_top_20_raion',
                  'sport_objects_raion', 'additional_education_raion',
                  'culture_objects_top_25', 'culture_objects_top_25_raion',
                  'shopping_centers_raion', 'office_raion', 'thermal_power_plant_raion',
                  'incineration_raion', 'oil_chemistry_raion', 'radiation_raion',
                  'railroad_terminal_raion', 'big_market_raion', 'nuclear_reactor_raion',
                  'detention_facility_raion', 'full_all', 'male_f', 'female_f', 'young_all',
                  'young_male', 'young_female', 'work_all', 'work_male', 'work_female',
                  'ekder_all', 'ekder_male', 'ekder_female', '0_6_all', '0_6_male',
                  '0_6_female', '7_14_all', '7_14_male', '7_14_female', '0_17_all',
                  '0_17_male', '0_17_female', '16_29_all', '16_29_male', '16_29_female',
                  '0_13_all', '0_13_male', '0_13_female', 'raion_build_count_with_material_info',
                  'build_count_block', 'build_count_wood', 'build_count_frame',
                  'build_count_brick', 'build_count_monolith', 'build_count_panel',
                  'build_count_foam', 'build_count_slag', 'build_count_mix',
                  'raion_build_count_with_builddate_info', 'build_count_before_1920',
                  'build_count_1921-1945', 'build_count_1946-1970', 'build_count_1971-1995',
                  'build_count_after_1995', 'ID_metro', 'metro_min_avto', 'metro_km_avto',
                  'metro_min_walk', 'metro_km_walk', 'kindergarten_km', 'school_km', 'park_km',
                  'green_zone_km', 'industrial_km', 'water_treatment_km', 'cemetery_km',
                  'incineration_km', 'railroad_station_walk_km', 'railroad_station_walk_min',
                  'ID_railroad_station_walk', 'railroad_station_avto_km', 'railroad_station_avto_min',
                  'ID_railroad_station_avto', 'public_transport_station_km',
                  'public_transport_station_min_walk', 'water_km', 'water_1line', 'mkad_km',
                  'ttk_km', 'sadovoe_km', 'bulvar_ring_km', 'big_road1_km', 'ID_big_road1',
                  'big_road1_1line', 'big_road2_km', 'ID_big_road2', 'railroad_km',
                  'railroad_1line', 'zd_vokzaly_avto_km', 'ID_railroad_terminal',
                  'bus_terminal_avto_km', 'ID_bus_terminal', 'oil_chemistry_km',
                  'nuclear_reactor_km', 'radiation_km', 'power_transmission_line_km',
                  'thermal_power_plant_km', 'ts_km', 'big_market_km', 'market_shop_km', 'fitness_km', 'swim_pool_km',
                  'ice_rink_km', 'stadium_km', 'basketball_km', 'hospice_morgue_km', 'detention_facility_km',
                  'public_healthcare_km', 'university_km', 'workplaces_km', 'shopping_centers_km',
                  'office_km', 'additional_education_km', 'preschool_km', 'big_church_km', 'church_synagogue_km',
                  'mosque_km', 'theater_km', 'museum_km', 'exhibition_km', 'catering_km', 'ecology', 'green_part_500',
                  'prom_part_500', 'office_count_500', 'office_sqm_500', 'trc_count_500', 'trc_sqm_500', 'cafe_count_500',
                  'cafe_sum_500_min_price_avg', 'cafe_sum_500_max_price_avg', 'cafe_avg_price_500',
                  'cafe_count_500_na_price', 'cafe_count_500_price_500', 'cafe_count_500_price_1000',
                  'cafe_count_500_price_1500', 'cafe_count_500_price_2500', 'cafe_count_500_price_4000',
                  'cafe_count_500_price_high', 'big_church_count_500', 'church_count_500', 'mosque_count_500',
                  'leisure_count_500', 'sport_count_500', 'market_count_500', 'green_part_1000', 'prom_part_1000',
                  'office_count_1000', 'office_sqm_1000', 'trc_count_1000', 'trc_sqm_1000', 'cafe_count_1000',
                  'cafe_sum_1000_min_price_avg', 'cafe_sum_1000_max_price_avg', 'cafe_avg_price_1000',
                  'cafe_count_1000_na_price', 'cafe_count_1000_price_500', 'cafe_count_1000_price_1000',
                  'cafe_count_1000_price_1500', 'cafe_count_1000_price_2500', 'cafe_count_1000_price_4000',
                  'cafe_count_1000_price_high', 'big_church_count_1000', 'church_count_1000', 'mosque_count_1000',
                  'leisure_count_1000', 'sport_count_1000', 'market_count_1000', 'green_part_1500', 'prom_part_1500',
                  'office_count_1500', 'office_sqm_1500', 'trc_count_1500', 'trc_sqm_1500', 'cafe_count_1500',
                  'cafe_sum_1500_min_price_avg', 'cafe_sum_1500_max_price_avg', 'cafe_avg_price_1500',
                  'cafe_count_1500_na_price', 'cafe_count_1500_price_500', 'cafe_count_1500_price_1000',
                  'cafe_count_1500_price_1500', 'cafe_count_1500_price_2500', 'cafe_count_1500_price_4000',
                  'cafe_count_1500_price_high', 'big_church_count_1500', 'church_count_1500', 'mosque_count_1500',
                  'leisure_count_1500', 'sport_count_1500', 'market_count_1500', 'green_part_2000', 'prom_part_2000',
                  'office_count_2000', 'office_sqm_2000', 'trc_count_2000', 'trc_sqm_2000', 'cafe_count_2000',
                  'cafe_sum_2000_min_price_avg', 'cafe_sum_2000_max_price_avg', 'cafe_avg_price_2000',
                  'cafe_count_2000_na_price', 'cafe_count_2000_price_500', 'cafe_count_2000_price_1000',
                  'cafe_count_2000_price_1500', 'cafe_count_2000_price_2500', 'cafe_count_2000_price_4000',
                  'cafe_count_2000_price_high', 'big_church_count_2000', 'church_count_2000', 'mosque_count_2000',
                  'leisure_count_2000', 'sport_count_2000', 'market_count_2000', 'green_part_3000', 'prom_part_3000',
                  'office_count_3000', 'office_sqm_3000', 'trc_count_3000', 'trc_sqm_3000', 'cafe_count_3000',
                  'cafe_sum_3000_min_price_avg', 'cafe_sum_3000_max_price_avg', 'cafe_avg_price_3000',
                  'cafe_count_3000_na_price', 'cafe_count_3000_price_500', 'cafe_count_3000_price_1000',
                  'cafe_count_3000_price_1500', 'cafe_count_3000_price_2500', 'cafe_count_3000_price_4000',
                  'cafe_count_3000_price_high', 'big_church_count_3000', 'church_count_3000', 'mosque_count_3000',
                  'leisure_count_3000', 'sport_count_3000', 'market_count_3000', 'green_part_5000', 'prom_part_5000',
                  'office_count_5000', 'office_sqm_5000', 'trc_count_5000', 'trc_sqm_5000', 'cafe_count_5000',
                  'cafe_sum_5000_min_price_avg', 'cafe_sum_5000_max_price_avg', 'cafe_avg_price_5000',
                  'cafe_count_5000_na_price', 'cafe_count_5000_price_500', 'cafe_count_5000_price_1000',
                  'cafe_count_5000_price_1500', 'cafe_count_5000_price_2500', 'cafe_count_5000_price_4000',
                  'cafe_count_5000_price_high', 'big_church_count_5000', 'church_count_5000', 'mosque_count_5000',
                  'leisure_count_5000', 'sport_count_5000', 'market_count_5000']:
            if k in df.columns:
                df.loc[df['kremlin_km'] < 0.1, k] = np.NAN

        df.loc[df['kremlin_km'] < 0.1, 'sub_area'] = 'ErrorRaionPoselenie'
        df.loc[df['kremlin_km'] < 0.1, 'kremlin_km'] = np.NAN

    # raw_data.insert(0, 'has_distance_error', 0)
    # raw_data.loc[raw_data['kremlin_km'] < 0.1, 'has_distance_error'] = 1


# fix_train_indexset = set(df_fixup[df_fixup.index < df_train.index.max()].index)
# bad_train_indexset = set(df_train[df_train.kremlin_km == df_train.kremlin_km.min()].index)
# unfixed_set = bad_train_indexset.difference(fix_train_indexset)
#
# print(len(unfixed_set))
# print(len(bad_train_indexset))
#
# train_df.drop(unfixed_set, inplace=True)
# df = pd.concat([train_df, test_df])
# for c in df_fixup.columns:
#     df.loc[df_fixup.index, c] = df_fixup[c]
# end bad data fix

# raw_data['sub_area'] = raw_data['sub_area'].map(
    #     {'Rostokino': 4480, 'Preobrazhenskoe': 4550, 'Timirjazevskoe': 4581, 'Lianozovo': 4330, 'Konkovo': 4830,
    #      'Kuncevo': 4682, 'PoselenieRjazanovskoe': 2315, 'PoselenieShherbinka': 2215, 'Vostochnoe': 3512,
    #      'PoselenieMihajlovoJarcevskoe': 2098, 'Pechatniki': 3980, 'Zamoskvoreche': 6600, 'Koptevo': 4409,
    #      'VostochnoeIzmajlovo': 3917, 'SokolinajaGora': 4452, 'Sokol': 4826, 'JuzhnoeTushino': 4316,
    #      'Brateevo': 3883, 'Jasenevo': 4375, 'Juzhnoportovoe': 4826, 'Dorogomilovo': 6440,
    #      'JuzhnoeMedvedkovo': 4240, 'Hamovniki': 7796, 'Tekstilshhiki': 3915, 'PoselenieMoskovskij': 2513,
    #      'SevernoeMedvedkovo': 4276, 'Babushkinskoe': 4686, 'Jaroslavskoe': 3694, 'NovoPeredelkino': 3566,
    #      'JuzhnoeButovo': 3714, 'Obruchevskoe': 4995, 'Strogino': 5068, 'OrehovoBorisovoSevernoe': 3802,
    #      'BirjulevoVostochnoe': 3438, 'ChertanovoSevernoe': 4320, 'PoselenieSosenskoe': 2590, 'Silino': 3351,
    #      'Lefortovo': 4630, 'NagatinoSadovniki': 4447, 'Matushkino': 3350, 'Molzhaninovskoe': 1374,
    #      'PoselenieShhapovskoe': 1341, 'Mitino': 4274, 'Bibirevo': 4116, 'PoseleniePervomajskoe': 1735,
    #      'Kurkino': 4276, 'Troickijokrug': 2336, 'Ljublino': 4066, 'Veshnjaki': 3730, 'Vojkovskoe': 4502,
    #      'ZapadnoeDegunino': 2898, 'Novogireevo': 4292, 'Metrogorodok': 4018, 'ChertanovoJuzhnoe': 4248,
    #      'Danilovskoe': 4758, 'Altufevskoe': 4068, 'Savelovskoe': 4886, 'Basmannoe': 5608, 'Mozhajskoe': 4151,
    #      'OrehovoBorisovoJuzhnoe': 4112, 'ProspektVernadskogo': 5542, 'Levoberezhnoe': 3993,
    #      'VostochnoeDegunino': 4024, 'Marino': 4152, 'Zjablikovo': 4203, 'Ivanovskoe': 3801, 'Rjazanskij': 3999,
    #      'SevernoeTushino': 4647, 'Vnukovo': 3313, 'Cheremushki': 5178, 'Lomonosovskoe': 5627,
    #      'HoroshevoMnevniki': 4827, 'Izmajlovo': 4282, 'Jakimanka': 5423, 'Solncevo': 3176,
    #      'PoselenieRogovskoe': 1255, 'Dmitrovskoe': 3868, 'Horoshevskoe': 5037, 'TeplyjStan': 4241,
    #      'Caricyno': 4174, 'Krasnoselskoe': 5860, 'PokrovskoeStreshnevo': 4612, 'Butyrskoe': 4517,
    #      'PoselenieKrasnopahorskoe': 1962, 'Begovoe': 5377, 'Krylatskoe': 5385, 'FiliDavydkovo': 4771,
    #      'Perovo': 3953, 'Sviblovo': 4310, 'PoselenieVoronovskoe': 2024, 'FilevskijPark': 4501,
    #      'PoselenieVnukovskoe': 2734, 'PoselenieDesjonovskoe': 2349, 'Alekseevskoe': 4971,
    #      'PoselenieNovofedorovskoe': 1247, 'Hovrino': 4233, 'Kotlovka': 4659, 'Kapotnja': 3514,
    #      'MarinaRoshha': 4698, 'PoselenieKokoshkino': 2214, 'NagatinskijZaton': 4162, 'Otradnoe': 4205,
    #      'SevernoeButovo': 4257, 'Novokosino': 3912, 'Savelki': 3375, 'Bogorodskoe': 4141, 'Taganskoe': 5522,
    #      'Marfino': 4216, 'Tverskoe': 3083, 'PoselenieVoskresenskoe': 2856, 'Beskudnikovskoe': 4041,
    #      'Akademicheskoe': 5114, 'Donskoe': 5585, 'Meshhanskoe': 6006, 'BirjulevoZapadnoe': 3509,
    #      'KosinoUhtomskoe': 3730, 'Nekrasovka': 2913, 'VyhinoZhulebino': 3887, 'Shhukino': 5078, 'Arbat': 7770,
    #      'Nizhegorodskoe': 4010, 'Presnenskoe': 5770, 'Golovinskoe': 4122, 'Zjuzino': 4696,
    #      'PoselenieKievskij': 2651, 'SevernoeIzmajlovo': 4067, 'PoselenieFilimonkovskoe': 1965, 'Sokolniki': 5777,
    #      'Nagornoe': 4623, 'OchakovoMatveevskoe': 4228, 'Severnoe': 3340, 'Ajeroport': 5041, 'Kuzminki': 4079,
    #      'Ostankinskoe': 5074, 'Gagarinskoe': 6011, 'StaroeKrjukovo': 3286, 'PoselenieMarushkinskoe': 2377,
    #      'Ramenki': 5171, 'MoskvorecheSaburovo': 4469, 'ChertanovoCentralnoe': 4463, 'Krjukovo': 3013,
    #      'Losinoostrovskoe': 4216, 'PoselenieMosrentgen': 2958, 'TroparevoNikulino': 4854, 'Goljanovo': 3972
    #  })


def correlation_matrix(df, labels):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Features Correlation')
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_yticklabels(labels, fontsize=10)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[.75, .8, .85, .90, .95, 1])
    plt.show()
