import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from scipy.special import stdtr
import scipy.stats as stats
import math as m

def my_stat_normality_multiple(data):
    total = len(data)
    i = 0
    for row in data:
        i += 1
        my_stat_normality(row, total, i)


def my_stat_normality(x, total=None, current=None):
    if total is None:
        total = 1

    if current is None:
        current = 1

    # mu, sigma = 0, .1 # s = np.random.normal(mu, sigma, 1000)

    mu = np.mean(x)
    sigma = np.std(x)

    print("Count samples: ", len(x), "; Median: ", mu, "; Sigma:", sigma)

    ## bins=np.arange(min(x), max(x) + binwidth, binwidth)
    r = stats.normaltest(x)
    # r = stats.shapiro(x)
    print(r)

    # H0: Variable A and Variable B are independent.
    # Ha: Variable A and Variable B are not independent.
    plot_normality(x, mu, sigma, len(x), total, current)


def plot_normality(data, mu, sigma, bins, total, current_index):
    # plt.subplot(3, total, 1 + (current_index - 1) * 3)
    plt.subplot(3, total, current_index)
    count, bins, ignored = plt.hist(data, bins, normed=True)

    plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) *
             np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)),
             linewidth=2, color='r')

    # plt.subplot(3, total, 2 + (current_index - 1) * 3)
    plt.subplot(3, total, current_index + total)
    stats.probplot(data, plot=plt)

    # plt.subplot(3, total, current_index * 3)
    plt.subplot(3, total, current_index + 2 * total)
    plt.boxplot(data, 0, '')

    if current_index == total:
        plt.show()


def compare_median(y, y2, show_graph=None):
    av = np.mean(y)
    av2 = np.mean(y2)
    std2 = np.std(y2)
    std = np.std(y)
    se = std2 / m.sqrt(len(y2))
    se_t = m.sqrt(m.pow(std, 2) / len(y) + m.pow(std2, 2) / len(y2))

    print("*\tFist set\tSecond set")
    print("Avg\t%s\t%s" % (round(av, 4), round(av2, 4)))
    print("Std\t%s\t\t%s" % (round(std, 5), round(std2, 5)))

    z = (av2 - av) / se
    t = (av2 - av) / se_t
    print()
    print("*\tNormal\t\tStudent")
    print("Se\t%s\t\t%s" % (round(se, 5), round(se_t, 5)))
    print("z/t\t%s\t%s" % (round(z, 5), round(t, 5)))
    ks = stats.ks_2samp(y, y2).pvalue
    print("Kolmogorov-Smirnov:\t\t%s" % ks)
    if ks > .05:
        if ks > .4:
            print("Looks like that sets are from the VERY SAME general distribution")
        else:
            print("Looks like that sets are from the SAME general distribution")
    else:
        if ks < .001:
            print("Looks like that sets are from the SIGNIFICANTLY DIFFERENT distributions")
        else:
            print("Looks like that sets are from the DIFFERENT distributions")

    if show_graph is not None:
        my_stat_normality_multiple([y, y2])


def correlate(x, y):
    # Use scipy.stats.ttest_ind.
    t, p = sc.stats.ttest_ind(x, y, equal_var=False)
    print("ttest_ind:            t = %g  p = %g" % (t, p))

    # Compute the descriptive statistics of a and b.
    abar = sc.mean(x)
    avar = sc.std(x)
    na = len(x)
    adof = na - 1

    bbar = sc.mean(y)
    bvar = sc.std(y)
    nb = len(y)
    bdof = nb - 1

    # Use scipy.stats.ttest_ind_from_stats.
    t2, p2 = sc.stats.ttest_ind_from_stats(abar, np.sqrt(avar), na,
                                           bbar, np.sqrt(bvar), nb,
                                           equal_var=False)
    print("ttest_ind_from_stats: t = %g  p = %g" % (t2, p2))

    # Use the formulas directly.
    tf = (abar - bbar) / np.sqrt(avar / na + bvar / nb)
    dof = (avar / na + bvar / nb) ** 2 / (avar ** 2 / (na ** 2 * adof) + bvar ** 2 / (nb ** 2 * bdof))
    pf = 2 * stdtr(dof, -np.abs(tf))

    print("formula:              t = %g  p = %g" % (tf, pf))


def fisher_criterion(v1, v2):
    return abs(np.mean(v1) - np.mean(v2)) / (np.var(v1) + np.var(v2))


# Assuming sample sizes are not equal, what test do I use
# to compare sample means under the following circumstances
# (please correct if any of the following are incorrect):
#
# Normal Distribution = True and Homogeneity of Variance = True
#
# scipy.stats.ttest_ind(sample_1, sample_2)
# Normal Distribution = True and Homogeneity of Variance = False
#
# scipy.stats.ttest_ind(sample_1, sample_2, equal_var = False)
# Normal Distribution = False and Homogeneity of Variance = True
#
# scipy.stats.mannwhitneyu(sample_1, sample_2)
# Normal Distribution = False and Homogeneity of Variance = False
