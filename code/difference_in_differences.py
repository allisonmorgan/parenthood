#!/usr/bin/env python
# coding: utf-8

from scripts import regression, plot_utils, cohort_utils
from scipy.stats import ttest_ind, mannwhitneyu, sem, ttest_1samp
from matplotlib import gridspec
from multiprocessing import Pool

import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import os

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Helvetica']


def compute_ci(estimates, alpha=0.05):
    coef_mean = estimates.mean()
    lower = estimates.quantile(q=alpha/2.0)
    upper = estimates.quantile(q=1.0-(alpha/2.0))

    return {'mean': coef_mean, 'lower': lower, 'upper': upper}


def get_params(res, param_name):
    est = res.params[param_name]
    upr = res.conf_int()[1][param_name]
    lwr = res.conf_int()[0][param_name]
    p_value = res.pvalues[param_name]
    return {'coef': est, 'upper': upr, 'lower': lwr, 'p_value': p_value}


def fit_model(formula, data):
    mod = smf.ols(formula=formula, data=data)
    res = mod.fit()
    return res


if __name__ == '__main__':
    # This script accepts a field and date as a command line argument
    parser = argparse.ArgumentParser()
    parser.add_argument("field", type=str, help="Field you want to consider")
    parser.add_argument("date", type=str, help="To append to frame file")
    parser.add_argument("ymax", type=str, help="Publications scale")
    args = parser.parse_args()
    print(args.field, args.date, args.ymax)

    FIELD = args.field
    DATE = args.date

    if FIELD in ["History", "Business"]:
        FILE_ENDING = 'raw'
    elif FIELD == 'CS':
        FILE_ENDING = 'adj'
    else:
        print("Incorrect field supplied.")
        exit()

    N_samples = 10
    y_max = int(args.ymax)

    color_mapping = {
        'CS': ('#5777D9', '#293866'),
        'Business': ('#CC3A35', '#661D1B'),
        'History': ('#8BCC60', '#466630')}

    plot_utils.ACCENT_COLOR, plot_utils.ALMOST_BLACK = color_mapping[FIELD]

    # Build model for men:
    gender = 'men'
    df_m_treated = pd.read_csv(
        '../data/treated/%s_publication_outcomes_%s_%s_%s.tsv' %
        (FILE_ENDING, gender, FIELD.lower(), DATE), sep='\t')
    df_m_control = pd.read_csv(
        '../data/control/%s_publication_outcomes_%s_%s_%s.tsv' %
        (FILE_ENDING, gender, FIELD.lower(), DATE), sep='\t')

    # Build model for women:
    gender = 'women'
    df_w_treated = pd.read_csv(
        '../data/treated/%s_publication_outcomes_%s_%s_%s.tsv' %
        (FILE_ENDING, gender, FIELD.lower(), DATE), sep='\t')
    df_w_control = pd.read_csv(
        '../data/control/%s_publication_outcomes_%s_%s_%s.tsv' %
        (FILE_ENDING, gender, FIELD.lower(), DATE), sep='\t')

    print(len(df_m_control['i'].unique()), len(df_m_treated['i'].unique()))
    print(len(df_w_control['i'].unique()), len(df_w_treated['i'].unique()))

    df_m_control['parent'] = False
    df_m_treated['parent'] = True
    df_m_control['gender'] = 'M'
    df_m_treated['gender'] = 'M'
    df_m_control['is_female'] = False
    df_m_treated['is_female'] = False

    # df_m_control['baby_pre_tenure'] = df_m_control.c < df_m_control.t
    # df_m_treated['baby_pre_tenure'] = df_m_treated.c < df_m_treated.t

    df_m_control['cumulative'] = df_m_control.groupby(['i', 'round'])['y'].cumsum()
    df_m_treated['cumulative'] = df_m_treated.groupby(['i', 'round'])['y'].cumsum()

    df_w_control['parent'] = False
    df_w_treated['parent'] = True
    df_w_control['gender'] = 'F'
    df_w_treated['gender'] = 'F'
    df_w_control['is_female'] = True
    df_w_treated['is_female'] = True

    # df_w_control['baby_pre_tenure'] = df_w_control.c < df_w_control.t
    # df_w_treated['baby_pre_tenure'] = df_w_treated.c < df_w_treated.t

    df_w_control['cumulative'] = df_w_control.groupby(['i', 'round'])['y'].cumsum()
    df_w_treated['cumulative'] = df_w_treated.groupby(['i', 'round'])['y'].cumsum()

    # Statistical Similarities between Control and Treatment Groups
    print("\nDifferences between parents / non-parents respect to \
           productivity?")
    parents = pd.concat([df_m_treated, df_w_treated])
    non_parents = pd.concat([df_m_control, df_w_control])

    parents_y = parents[parents.t == 0]['y'].dropna()
    not_parents_y = non_parents[(non_parents.t == 0)]['y'].dropna()

    print(ttest_ind(parents_y, not_parents_y, equal_var=False))

    parents_sem = parents_y.var()/len(parents_y)
    not_parents_sem = not_parents_y.var()/len(not_parents_y)

    num = (parents_sem + not_parents_sem)**2
    denom = (parents_sem**2)/(len(parents_y)-1) + \
            (not_parents_sem**2)/(len(not_parents_y)-1)
    dof = num/denom
    print(len(parents_y), len(not_parents_y), dof)

    print((parents_y.mean(), parents_y.std()),
          (not_parents_y.mean(), not_parents_y.std()))

    print("\nDifferences between fathers / non-fathers respect to \
           productivity?")
    fathers_y = df_m_treated[df_m_treated.t == 0]['y'].dropna()
    non_fathers_y = df_m_control[(df_m_control.t == 0)]['y'].dropna()

    print(ttest_ind(fathers_y, non_fathers_y, equal_var=False))

    fathers_sem = fathers_y.var()/len(fathers_y)
    not_fathers_sem = non_fathers_y.var()/len(non_fathers_y)

    num = (fathers_sem + not_fathers_sem)**2
    denom = (fathers_sem**2)/(len(fathers_y)-1) + \
            (not_fathers_sem**2)/(len(non_fathers_y)-1)
    dof = num/denom
    print(len(fathers_y), len(non_fathers_y), dof)

    print((fathers_y.mean(), fathers_y.std()),
          (non_fathers_y.mean(), non_fathers_y.std()))

    print("\nDifferences between mothers / non-mothers respect to \
          productivity?")
    mothers_y = df_w_treated[df_w_treated.t == 0]['y'].dropna()
    not_mothers_y = df_w_control[(df_w_control.t == 0)]['y'].dropna()

    print(ttest_ind(mothers_y, not_mothers_y, equal_var=False))
    mothers_sem = mothers_y.var()/len(mothers_y)
    not_mothers_sem = not_mothers_y.var()/len(not_mothers_y)

    num = (mothers_sem + not_mothers_sem)**2
    denom = (mothers_sem**2)/(len(mothers_y)-1) + \
            (not_mothers_sem**2)/(len(not_mothers_y)-1)
    dof = num/denom
    print(len(mothers_y), len(not_mothers_y), dof)

    print((mothers_y.mean(), mothers_y.std()),
          (not_mothers_y.mean(), not_mothers_y.std()))

    print("\nFATHERS")
    print("Indistinguishable with respect to age at first birth?")
    if ('age' in df_m_control.columns) and ('age' in df_m_treated.columns):
        print(ttest_ind(
          df_m_control[df_m_control.t == 0]['age'],
          df_m_treated[df_m_treated.t == 0]['age'],
          equal_var=False),
          ((df_m_control[df_m_control.t == 0]['age'].mean(),
            df_m_control[df_m_control.t == 0]['age'].std()),
           (df_m_treated[df_m_treated.t == 0]['age'].mean(),
            df_m_treated[df_m_treated.t == 0]['age'].std())))

    # print("Indistinguishable with respect to career age at first birth?")
    # print(ttest_ind(
    #   df_m_control[df_m_control.t == 0]['c'],
    #   df_m_treated[df_m_treated.t == 0]['c'],
    #   equal_var=False),
    #   ((df_m_control[df_m_control.t == 0]['c'].mean(),
    #     df_m_control[df_m_control.t == 0]['c'].std()),
    #    (df_m_treated[df_m_treated.t == 0]['c'].mean(),
    #     df_m_treated[df_m_treated.t == 0]['c'].std())))

    print("Indistinguishable with respect to prestige?")
    print(mannwhitneyu(
      df_m_control[(df_m_control['round'] == 0) & (df_m_control.t == 0)]['pi'],
      df_m_treated[df_m_treated.t == 0]['pi']),
      ((df_m_control[(df_m_control['round'] == 0) &
                     (df_m_control.t == 0)]['pi'].mean(),
        df_m_control[(df_m_control['round'] == 0) &
                     (df_m_control.t == 0)]['pi'].std()),
       (df_m_treated[df_m_treated.t == 0]['pi'].mean(),
        df_m_treated[df_m_treated.t == 0]['pi'].std())))

    print("\nMOTHERS")
    print("Indistinguishable with respect to age at first birth?")
    if ('age' in df_w_control.columns) and ('age' in df_w_treated.columns):
        print(ttest_ind(
          df_w_treated[df_w_treated.t == 0]['age'],
          df_w_control[df_w_control.t == 0]['age'],
          equal_var=False),
          ((df_w_control[df_w_control.t == 0]['age'].mean(),
            df_w_control[df_w_control.t == 0]['age'].std()),
           (df_w_treated[df_w_treated.t == 0]['age'].mean(),
            df_w_treated[df_w_treated.t == 0]['age'].std())))

    # print("Indistinguishable with respect to career age at first birth?")
    # print(ttest_ind(
    #   df_w_treated[df_w_treated.t == 0]['c'],
    #   df_w_control[df_w_control.t == 0]['c'],
    #   equal_var=False),
    #   ((df_w_control[df_w_control.t == 0]['c'].mean(),
    #     df_w_control[df_w_control.t == 0]['c'].std()),
    #    (df_w_treated[df_w_treated.t == 0]['c'].mean(),
    #     df_w_treated[df_w_treated.t == 0]['c'].std())))

    print("Indistinguishable with respect to prestige?")
    print(mannwhitneyu(
      df_w_control[df_w_control.t == 0]['pi'],
      df_w_treated[df_w_treated.t == 0]['pi']),
      ((df_w_control[df_w_control.t == 0]['pi'].mean(),
        df_w_control[df_w_control.t == 0]['pi'].std()),
       (df_w_treated[df_w_treated.t == 0]['pi'].mean(),
        df_w_treated[df_w_treated.t == 0]['pi'].std())))

    # ### Parallel Trends
    print("\nParallel trends?")
    pre_times = [-5, -4, -3, -2]

    pre_trend_slopes = df_w_control[
      df_w_control.t.isin(pre_times)].groupby(['i'])['y'].mean()
    print('Women w/o kids:\t', np.nanmean(pre_trend_slopes),
          np.nanstd(pre_trend_slopes))

    pre_trend_slopes = df_w_treated[
      df_w_treated.t.isin(pre_times)].groupby(['i'])['y'].mean()
    print('Women w/ kids:\t', np.nanmean(pre_trend_slopes),
          np.nanstd(pre_trend_slopes))
    print(ttest_ind(
      df_w_treated[df_w_treated.t.isin(pre_times)].groupby(['i'])['y'].mean(),
      df_w_control[df_w_control.t.isin(pre_times)].groupby(['i'])['y'].mean()))

    pre_trend_slopes = df_m_control[
      df_m_control.t.isin(pre_times)].groupby(['i'])['y'].mean()
    print('\nMen w/o kids:\t',
          np.nanmean(pre_trend_slopes),
          np.nanstd(pre_trend_slopes))

    pre_trend_slopes = df_m_treated[
      df_m_treated.t.isin(pre_times)].groupby(['i'])['y'].mean()
    print('Men w/ kids:\t',
          np.nanmean(pre_trend_slopes),
          np.nanstd(pre_trend_slopes))
    print(ttest_ind(
      df_m_treated[df_m_treated.t.isin(pre_times)].groupby(['i'])['y'].mean(),
      df_m_control[df_m_control.t.isin(pre_times)].groupby(['i'])['y'].mean()))

    print("\nTrends in the difference in difference plots")

    # Split the difference in time for each
    mid_m = 2000
    ids_m_pre = df_m_treated.loc[(df_m_treated['t'] == 0) &
                                 df_m_treated.pre_2000]['i'].unique()
    ids_m_post = df_m_treated.loc[(df_m_treated['t'] == 0) &
                                  (~df_m_treated.pre_2000)]['i'].unique()

    mid_w = 2000
    ids_w_pre = df_w_treated.loc[(df_w_treated['t'] == 0) &
                                 (df_w_treated.pre_2000)]['i'].unique()
    ids_w_post = df_w_treated.loc[(df_w_treated['t'] == 0) &
                                  (~df_w_treated.pre_2000)]['i'].unique()

    print((len(ids_m_pre), len(ids_w_pre)), (len(ids_m_post), len(ids_w_post)))
    print(len(ids_m_pre) + len(ids_w_pre) + len(ids_m_post) + len(ids_w_post))

    younger_w = df_w_treated.loc[df_w_treated['i'].isin(ids_w_post)]
    younger_m = df_m_treated.loc[df_m_treated['i'].isin(ids_m_post)]

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(4, 4), sharey=True)

    wtrend = cohort_utils.generate_average_cumulative(younger_w).tolist()
    std = regression.get_bootstrap_trajectories(younger_w, N_samples)
    ax.plot(regression.T, wtrend, label='Mothers', linestyle='-', marker='o',
            color=plot_utils.ACCENT_COLOR)
    ax.fill_between(regression.T, wtrend-2*std, wtrend+2*std,
                    color=plot_utils.ACCENT_COLOR, alpha=0.2)

    mtrend = cohort_utils.generate_average_cumulative(younger_m).tolist()
    std = regression.get_bootstrap_trajectories(younger_m, N_samples)
    ax.plot(regression.T, mtrend, label='Fathers', linestyle='-', marker='o',
            color=plot_utils.ALMOST_BLACK)
    ax.fill_between(regression.T, mtrend-2*std, mtrend+2*std,
                    color=plot_utils.ALMOST_BLACK, alpha=0.2)
    ax.set_ylabel('Number of Publications', fontsize=plot_utils.LABEL_SIZE)
    ax.set_xlim(-6, 11)
    ax.set_xticks(range(-5, 11, 5))

    ax.annotate('%d' % round((mtrend[-1] - wtrend[-1])),
                xy=(10, wtrend[-1]),
                xytext=(12, wtrend[-1] + 0.5*(mtrend[-1] - wtrend[-1])),
                arrowprops=dict(arrowstyle="-", connectionstyle="arc3",
                                color='0.125'),
                color=plot_utils.LIGHT_COLOR)
    ax.annotate('%d' % round((mtrend[-1] - wtrend[-1])),
                xy=(10, mtrend[-1]),
                xytext=(12, wtrend[-1] + 0.5*(mtrend[-1] - wtrend[-1])),
                arrowprops=dict(arrowstyle="-", connectionstyle="arc3",
                                color='0.125'),
                color='0.125')

    # Add text
    print("Post-2000")
    print('Long-run W/M (Parents)',
          (wtrend[-1]/mtrend[-1]), mtrend[-1]-wtrend[-1])
    print('Men: ', mtrend[-1], 'Women: ', wtrend[-1])

    # Calculate the number of years missing for mothers to reach fathers
    # (relative to their child's birth)
    post_baby_men = younger_m[younger_m.t > 0]
    post_baby_women = younger_w[younger_w.t > 0]

    productivity_m = cohort_utils.generate_average_cumulative(post_baby_men)
    productivity_w = cohort_utils.generate_average_cumulative(post_baby_women)

    print('Lost years?', (productivity_m[10]-productivity_w[10]),
          (productivity_m[10] - productivity_w[10]) /
          (productivity_w[10]/len(productivity_w)),
          len(productivity_w))

    ax.set_xlabel("Time Relative to\nFirst Child's Birth (Years)",
                  fontsize=plot_utils.LABEL_SIZE)

    #
    older_w = df_w_treated.loc[df_w_treated['i'].isin(ids_w_pre)]
    older_m = df_m_treated.loc[df_m_treated['i'].isin(ids_m_pre)]

    ins = ax.inset_axes([0.15, 0.5, 0.3, 0.5])
    wtrend = cohort_utils.generate_average_cumulative(older_w).tolist()
    std = regression.get_bootstrap_trajectories(older_w, N_samples)
    ins.plot(regression.T, wtrend, label='Mothers', linestyle='-', marker='o',
             color=plot_utils.ACCENT_COLOR,
             markersize=2, linewidth=1)
    ins.fill_between(regression.T, wtrend-2*std, wtrend+2*std,
                     color=plot_utils.ACCENT_COLOR, alpha=0.2)

    mtrend = cohort_utils.generate_average_cumulative(older_m).tolist()
    std = regression.get_bootstrap_trajectories(older_m, N_samples)
    ins.plot(regression.T, mtrend, label='Fathers', linestyle='-', marker='o',
             color=plot_utils.ALMOST_BLACK, markersize=2, linewidth=1)
    ins.fill_between(regression.T, mtrend-2*std, mtrend+2*std,
                     color=plot_utils.ALMOST_BLACK, alpha=0.2)

    ins.annotate('%d' % round((mtrend[-1] - wtrend[-1])),
                 xy=(10, wtrend[-1]),
                 xytext=(12, wtrend[-1] + 0.5*(mtrend[-1] - wtrend[-1])),
                 arrowprops=dict(arrowstyle="-", connectionstyle="arc3",
                                 color='0.125'),
                 color=plot_utils.LIGHT_COLOR)
    ins.annotate('%d' % round((mtrend[-1] - wtrend[-1])),
                 xy=(10, mtrend[-1]),
                 xytext=(12, wtrend[-1] + 0.5*(mtrend[-1] - wtrend[-1])),
                 arrowprops=dict(arrowstyle="-", connectionstyle="arc3",
                                 color='0.125'),
                 color='0.125')

    # Add text
    print("Pre-2000")
    print('Long-run W/M (Parents)',
          (wtrend[-1]/mtrend[-1]), mtrend[-1]-wtrend[-1])
    print('Men: ', mtrend[-1], 'Women: ', wtrend[-1])

    # Calculate the number of years missing for mothers to reach fathers
    # (relative to their child's birth)
    pre_baby_men = older_m[older_m.t > 0]
    pre_baby_women = older_w[older_w.t > 0]

    productivity_m = cohort_utils.generate_average_cumulative(pre_baby_men)
    productivity_w = cohort_utils.generate_average_cumulative(pre_baby_women)

    print('Lost years?', (productivity_m[10]-productivity_w[10]),
          (productivity_m[10]-productivity_w[10]) /
          (productivity_w[10]/len(productivity_w)),
          len(productivity_w))

    ins.set_xlim(-6, 11)
    ins.set_xticks(range(-5, 11, 5))

    plot_utils.finalize(ax)
    plot_utils.finalize(ins)

    ins.tick_params(labelsize=9)

    plt.tight_layout()
    plt.savefig(
      '../plots/diff_in_diff/productivity_dd_w_kids_%s_%s_trend_%s.pdf'
      % (FILE_ENDING, FIELD.lower(), DATE), dpi=1000)

    print("\nComparative interrupted time series")

    # fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    # df_w.plot(x='c', y='t', kind='scatter', ax=axs[0],  alpha=0.3)

    # df_w_temp = df_w.copy()
    # def tt_track_mapping(x):
    #     if x < 0:
    #         return 'pre-tt'
    #     if x >= 0 and x <= 5:
    #         return 'early-tt'
    #     if x > 5:
    #         return 'post-tt'

    # df_w_temp['stage'] = df_w_temp['c'].apply(tt_track_mapping)
    # df_w_temp.boxplot(by='stage', column='t', ax=axs[1])

    # from scipy.stats import pearsonr
    # print("Correlation between career age and parenthood:\t",
    #       pearsonr(df_w.t, df_w.c))

    # F, p = stats.f_oneway(df_w_temp[df_w_temp.stage=='pre-tt'].t,
    #                       df_w_temp[df_w_temp.stage=='early-tt'].t,
    #                       df_w_temp[df_w_temp.stage=='post-tt'].t)

    # formula = 'y ~ t + C(t>0) + t:C(t>0) + C(parent) + C(parent):t + \
    #            C(parent):C(t>0) + C(parent):C(t>0):t + pi + \
    #            C((c < 5) & (c >= 0)) + C(c < 0) + C(c > 5)'
    # men_data = pd.concat([df_m_control, df_m_treated])
    # women_data = pd.concat([df_w_control, df_w_treated])

    # # Fit model for women
    # mod = smf.ols(formula=formula, data=pd.concat([men_data]))
    # res = mod.fit()
    # # res.summary()

    # formula = 'y ~ t + C(t>0) + t:C(t>0) + C(parent) + C(parent):t + \
    #            C(parent):C(t>0) + C(parent):C(t>0):t + pi'

    # mod_less = smf.ols(formula=formula, data=pd.concat([women_data]))
    # res_less = mod_less.fit()
    # print(res_less.summary())

    # print(sm.stats.anova_lm(res_less, res))

    women_params = []
    men_params = []
    women_results = []
    men_results = []
    women_shock_estimates = []
    men_shock_estimates = []
    women_slope_estimates = []
    men_slope_estimates = []
    women_expectations = []
    men_expectations = []

    print("Modeling men with children versus men without children")

    formula = 'y ~ t + C(t>0) + t:C(t>0) + C(parent) + C(parent):t + \
                  C(parent):C(t>0) + C(parent):C(t>0):t + pi'

    pool = Pool(os.cpu_count() - 1)
    multiple_results = []
    for iteration in df_m_control['round'].unique():
        men_data = pd.concat([
          df_m_control[df_m_control['round'] == iteration], df_m_treated])
        multiple_results.append(pool.apply_async(fit_model, (formula, men_data)))

    print("Loaded tasks asynchronously")

    for result in multiple_results:
        men_res = result.get(timeout=None)

        # Get all the parameters of the model fit
        men_params.append(men_res.params)
        men_results.append(men_res)

        # Get the shock andd slope estimates for men and women
        men_shock_estimates.append(
          get_params(men_res, 'C(parent)[T.True]:C(t > 0)[T.True]'))
        men_slope_estimates.append(
          get_params(men_res, 'C(parent)[T.True]:C(t > 0)[T.True]:t'))

        # Get expected value at t = 0 (among parents)
        men_expectations.append(
          np.mean(men_res.predict(men_data[(men_data.t == 0) & men_data.parent])))

    pool.close()

    parameter_names = {'beta_0': 'Intercept', 'beta_1': 't',
                       'beta_2': 'C(t > 0)[T.True]',
                       'beta_3': 't:C(t > 0)[T.True]',
                       'beta_4': 'C(parent)[T.True]',
                       'beta_5': 'C(parent)[T.True]:t',
                       'beta_6': 'C(parent)[T.True]:C(t > 0)[T.True]',
                       'beta_7': 'C(parent)[T.True]:C(t > 0)[T.True]:t',
                       'beta_8': 'pi'}

    for coef_short, coef_long in parameter_names.items():
        print(coef_short)
        temp = [get_params(men_res, coef_long)['coef'] for men_res in men_results]
        print('%.3f (std: %.3f, sem: %.3f, CI: [%.3f, %.3f])' % (np.mean(temp),
              np.std(temp), sem(temp), compute_ci(pd.Series(temp))['lower'],
              compute_ci(pd.Series(temp))['upper']))
        print(ttest_1samp(temp, popmean=0))

    print("\nModeling women with children versus women without children")

    pool = Pool(os.cpu_count() - 1)
    multiple_results = []

    for iteration in df_w_control['round'].unique():
        women_data = pd.concat(
          [df_w_control[df_w_control['round'] == iteration], df_w_treated])
        multiple_results.append(pool.apply_async(fit_model, (formula, women_data)))

    print("Loaded tasks asynchronously")

    for result in multiple_results:
        women_res = result.get(timeout=None)

        women_params.append(women_res.params)
        women_results.append(women_res)

        # Get the shock andd slope estimates for men and women
        women_shock_estimates.append(
          get_params(women_res, 'C(parent)[T.True]:C(t > 0)[T.True]'))
        women_slope_estimates.append(
          get_params(women_res, 'C(parent)[T.True]:C(t > 0)[T.True]:t'))

        # Get expected value at t = 0 (among parents)
        women_expectations.append(np.mean(
          women_res.predict(women_data[(women_data.t == 0) & women_data.parent])))

    for coef_short, coef_long in parameter_names.items():
        print(coef_short)
        temp = [get_params(women_res, coef_long)['coef'] for
                women_res in women_results]
        print('%.3f (std: %.3f, sem: %.3f, CI: [%.3f, %.3f])' %
              (np.mean(temp), np.std(temp), sem(temp),
               compute_ci(pd.Series(temp))['lower'],
               compute_ci(pd.Series(temp))['upper']))
        print(ttest_1samp(temp, popmean=0))

    pool.close()

    print("\nModeling women without children versus men without children")

    pool = Pool(os.cpu_count() - 1)
    multiple_results = []

    not_parents_shock_estimates = []
    not_parents_slope_estimates = []
    not_parents_expectations = []
    not_parents_results = []

    formula = 'y ~ t + C(t > 0) + t:C(t > 0) + C(is_female) + C(is_female):t + \
              C(is_female):C(t > 0) + C(is_female):C(t > 0):t + pi'

    for iteration in df_m_control['round'].unique():
        not_parents_data = pd.concat(
          [df_m_control[df_m_control['round'] == iteration],
           df_w_control[df_w_control['round'] == iteration]])
        multiple_results.append(pool.apply_async(
          fit_model, (formula, not_parents_data)))

    print("Loaded tasks asynchronously")
    for result in multiple_results:
        not_parents_res = result.get(timeout=None)

        not_parents_results.append(not_parents_res)

        not_parents_shock_estimates.append(
          get_params(not_parents_res, 'C(is_female)[T.True]:C(t > 0)[T.True]'))
        not_parents_slope_estimates.append(
          get_params(not_parents_res, 'C(is_female)[T.True]:C(t > 0)[T.True]:t'))

        # Expected value at t = 0 among women
        not_parents_expectations.append(np.mean(not_parents_res.predict(
            not_parents_data[(not_parents_data.t == 0) &
                             not_parents_data.is_female])))

    parameter_names = {'beta_0': 'Intercept', 'beta_1': 't',
                       'beta_2': 'C(t > 0)[T.True]',
                       'beta_3': 't:C(t > 0)[T.True]',
                       'beta_4': 'C(is_female)[T.True]',
                       'beta_5': 'C(is_female)[T.True]:t',
                       'beta_6': 'C(is_female)[T.True]:C(t > 0)[T.True]',
                       'beta_7': 'C(is_female)[T.True]:C(t > 0)[T.True]:t',
                       'beta_8': 'pi'}

    for coef_short, coef_long in parameter_names.items():
        print(coef_short)
        temp = [get_params(not_parents_res, coef_long)['coef'] for
                not_parents_res in not_parents_results]
        print('%.3f (std: %.3f, sem: %.3f, CI: [%.3f, %.3f])' %
              (np.mean(temp), np.std(temp), sem(temp),
               compute_ci(pd.Series(temp))['lower'],
               compute_ci(pd.Series(temp))['upper']))
        print(ttest_1samp(temp, popmean=0))

    pool.close()

    print("\nModeling women with children versus men with children")

    parents_shock_estimates = []
    parents_slope_estimates = []
    parents_expectations = []

    parents_data = pd.concat([df_m_treated, df_w_treated])

    parents_mod = smf.ols(formula=formula, data=parents_data)
    parents_res = parents_mod.fit()

    parents_shock_estimates.append(
      get_params(parents_res, 'C(is_female)[T.True]:C(t > 0)[T.True]'))
    parents_slope_estimates.append(
      get_params(parents_res, 'C(is_female)[T.True]:C(t > 0)[T.True]:t'))

    parents_expectations.append(np.mean(parents_res.predict(
      parents_data[(parents_data.t == 0) & parents_data.is_female])))

    for coef_short, coef_long in parameter_names.items():
        print(coef_short)
        temp = get_params(parents_res, coef_long)
        mean, lower, upper = temp['coef'], temp['lower'], temp['upper']
        p_value = temp['p_value']
        sem = parents_res.bse[coef_long]
        print('%.3f (sem: %.3f, CI: [%.3f, %.3f])' % (mean, sem, lower, upper))
        print(p_value)

    #
    # Plot men's average productivity with and without kids
    #
    fig, ax = plt.subplots(ncols=2, nrows=1,
                           figsize=plot_utils.DOUBLE_FIG_SIZE, sharey=True)

    trend = np.array([df_m_control[df_m_control['t'] == t]['y'].mean(skipna=True)
                      for t in regression.T])
    std = regression.get_bootstrap_trajectories(
      df_m_control, N_samples, cumulative=False)

    ax[0].plot(regression.T, trend, label='w/o Children', linestyle='dotted',
               marker='o', markerfacecolor='white',
               color=plot_utils.ALMOST_BLACK)
    ax[0].fill_between(regression.T, trend-2*std, trend+2*std,
                       color=plot_utils.ALMOST_BLACK, alpha=0.2)

    adjusted_trend = np.array(
      [df_m_treated[df_m_treated['t'] == t]['y'].mean(skipna=True)
       for t in regression.T])
    std = regression.get_bootstrap_trajectories(df_m_treated, N_samples,
                                                cumulative=False)

    ax[0].plot(regression.T, adjusted_trend, label='w/ Children',
               linestyle='-', marker='o', color=plot_utils.ALMOST_BLACK)
    ax[0].fill_between(regression.T, adjusted_trend-2*std, adjusted_trend+2*std,
                       color=plot_utils.ALMOST_BLACK, alpha=0.2)
    ax[0].axvline(x=0, ls=":")

    # Add text
    ax[0].set_title("Average Annual Productivity (Men)",
                    fontsize=plot_utils.TITLE_SIZE)
    ax[0].set_xlabel("Time Relative to First Child's Birth (Years)")
    ax[0].set_ylabel("Number of Publications")
    ax[0].legend(loc='upper left', fontsize=plot_utils.LEGEND_SIZE,
                 frameon=False, ncol=1)

    ax[0].text(0.95, 0.2, r'$\hat{\beta}_{6}$: %.2f [%.2f, %.2f]' %
               tuple(compute_ci(pd.DataFrame(men_shock_estimates).coef).values()),
               ha='right', va='center', transform=ax[0].transAxes,
               fontsize=plot_utils.LEGEND_SIZE)
    ax[0].text(0.95, 0.1, r'$\hat{\beta}_{7}$: %.2f [%.2f, %.2f]' %
               tuple(compute_ci(pd.DataFrame(men_slope_estimates).coef).values()),
               ha='right', va='center', transform=ax[0].transAxes,
               fontsize=plot_utils.LEGEND_SIZE)

    ax[0].set_xticks(range(-5, 11))
    # Add styling and save
    plot_utils.finalize(ax[0])

    # Plot women's average productivity with and without kids
    trend = np.array([df_w_control[df_w_control['t'] == t]['y'].mean(skipna=True)
                      for t in regression.T])

    std = regression.get_bootstrap_trajectories(df_w_control, N_samples,
                                                cumulative=False)
    ax[1].plot(regression.T, trend, label='w/o Children', linestyle='dotted',
               marker='o', markerfacecolor='white',
               color=plot_utils.ACCENT_COLOR)
    ax[1].fill_between(regression.T, trend-2*std, trend+2*std,
                       color=plot_utils.ACCENT_COLOR, alpha=0.2)

    adjusted_trend = np.array(
      [df_w_treated[df_w_treated['t'] == t]['y'].mean(skipna=True)
       for t in regression.T])
    std = regression.get_bootstrap_trajectories(
      df_w_treated, N_samples, cumulative=False)

    ax[1].plot(regression.T, adjusted_trend, label='w/ Children',
               linestyle='-', marker='o', color=plot_utils.ACCENT_COLOR)
    ax[1].fill_between(regression.T, adjusted_trend-2*std, adjusted_trend+2*std,
                       color=plot_utils.ACCENT_COLOR, alpha=0.2)
    ax[1].axvline(x=0, ls=":")
    # Add text
    ax[1].set_title("Average Annual Productivity (Women)",
                    fontsize=plot_utils.TITLE_SIZE)
    ax[1].set_xlabel("Time Relative to First Child's Birth (Years)")

    ax[1].text(
      0.95, 0.2, r'$\hat{\beta}_{6}$: %.2f [%.2f, %.2f]' %
      tuple(compute_ci(pd.DataFrame(women_shock_estimates).coef).values()),
      ha='right', va='center', transform=ax[1].transAxes,
      fontsize=plot_utils.LEGEND_SIZE)
    ax[1].text(
      0.95, 0.1, r'$\hat{\beta}_{7}$: %.2f [%.2f, %.2f]' %
      tuple(compute_ci(pd.DataFrame(women_slope_estimates).coef).values()),
      ha='right', va='center', transform=ax[1].transAxes,
      fontsize=plot_utils.LEGEND_SIZE)

    ax[1].set_xticks(range(-5, 11))
    ax[1].set_ylim(-0.2, y_max+.2)

    # Add styling and save
    plot_utils.finalize(ax[1])
    plt.tight_layout()
    plt.savefig(
      '../plots/diff_in_diff/annual_productivity_dd_%s_%s_kid_%s.pdf' %
      (FILE_ENDING, FIELD.lower(), DATE), dpi=500)

    print("Expected values under model:")
    print({
        'men_w_kid': np.mean(men_expectations),
        'women_w_kid_vs_women_wo_kid': np.mean(women_expectations),
        'women_w_kid_vs_men_w_kid': np.mean(parents_expectations),
        'women_wo_kid': np.mean(not_parents_expectations)
    })

    print("Percentage changes to productivity (immediately):")
    estimates = {
      'men_w_kid_vs_men_wo_kid':
      compute_ci(pd.DataFrame(men_shock_estimates).coef)['mean']/np.mean(men_expectations),
      'women_w_kid_vs_women_wo_kid':
      compute_ci(pd.DataFrame(women_shock_estimates).coef)['mean']/np.mean(women_expectations),
      'women_w_kid_vs_men_w_kid':
      compute_ci(pd.DataFrame(parents_shock_estimates).coef)['mean']/np.mean(parents_expectations),
      'women_wo_kid_vs_men_wo_kid':
      compute_ci(pd.DataFrame(not_parents_shock_estimates).coef)['mean']/np.mean(not_parents_expectations)
    }

    fig = plt.figure(figsize=(7.25, 3.25), dpi=1000)
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1])
    ax = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    # Men without kids
    trend = np.array([df_m_control[df_m_control['t'] == t]['y'].mean(skipna=True)
                      for t in regression.T])
    ax.plot(regression.T[:5], trend[:5], marker='o',
            markerfacecolor='w', color=plot_utils.ALMOST_BLACK,
            linestyle='dotted', clip_on=False, zorder=1e2)
    ax.plot(regression.T[6:], trend[6:], linestyle='dotted', marker='o',
            markerfacecolor='w', color=plot_utils.ALMOST_BLACK, clip_on=False,
            zorder=1e2)

    # Men with kids
    adjusted_trend = np.array(
      [df_m_treated[df_m_treated['t'] == t]['y'].mean(skipna=True)
       for t in regression.T])
    ax.plot(regression.T[:5], adjusted_trend[:5], linestyle='-', marker='o',
            color=plot_utils.ALMOST_BLACK, clip_on=False, zorder=1e2)
    ax.plot(regression.T[6:], adjusted_trend[6:], linestyle='-', marker='o',
            color=plot_utils.ALMOST_BLACK, clip_on=False, zorder=1e2)

    # Women without kids
    trend = np.array([df_w_control[df_w_control['t'] == t]['y'].mean(skipna=True)
                      for t in regression.T])
    ax.plot(regression.T[:5], trend[:5], linestyle='dotted', marker='o',
            markerfacecolor='w', color=plot_utils.ACCENT_COLOR, clip_on=False,
            zorder=1e2,)
    ax.plot(regression.T[6:], trend[6:], linestyle='dotted', marker='o',
            markerfacecolor='w', color=plot_utils.ACCENT_COLOR, clip_on=False,
            zorder=1e2)

    # Women without kids
    adjusted_trend = np.array(
      [df_w_treated[df_w_treated['t'] == t]['y'].mean(skipna=True)
       for t in regression.T])
    ax.plot(regression.T[:5], adjusted_trend[:5], marker='o',
            color=plot_utils.ACCENT_COLOR, linestyle='-', clip_on=False,
            zorder=1e2)
    ax.plot(regression.T[6:], adjusted_trend[6:], linestyle='-', marker='o',
            color=plot_utils.ACCENT_COLOR, clip_on=False, zorder=1e2)

    ax.plot([-1], [-1], ':o', color=plot_utils.LIGHT_COLOR,
            markerfacecolor='w', label="w/o Children")
    ax.plot([-1], [-1], '-o', color=plot_utils.LIGHT_COLOR,
            label="w/ Children")
    ax.plot([-1], [-1], 'o', color=plot_utils.ACCENT_COLOR,
            label="Women")
    ax.plot([-1], [-1], 'o', color=plot_utils.ALMOST_BLACK,
            label="Men")

    ax.set_xticks(range(-5, 11))
    ax.set_xlim(-5, 10)

    ax.set_yticks(range(1, y_max+1))
    ax.set_ylim(0, y_max-1)

    ax.set_ylabel("Avg. Annual Productivity", labelpad=97)
    ax.set_xlabel("Time Relative to First Child's Birth (Years)")
    ax.legend(loc=(0.001, 0.65), frameon=False)

    # Move the y-axis through (0, 0)
    ax.spines['left'].set_position('zero')

    # CUSTOM LEGEND SUUUUP
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    keys = ['men_w_kid_vs_men_wo_kid', 'women_w_kid_vs_women_wo_kid',
            'women_w_kid_vs_men_w_kid', 'women_wo_kid_vs_men_wo_kid']
    labels = ['Men with and without children',
              'Women with and without children',
              'Women and men with children',
              'Women and men without children']
    vals = [estimates[k] for k in keys]
    markerfacecolors = [[plot_utils.ALMOST_BLACK, 'w'],
                        [plot_utils.ACCENT_COLOR, 'w'],
                        [plot_utils.ACCENT_COLOR, plot_utils.ALMOST_BLACK],
                        ['w', 'w']]
    line_colors = [[plot_utils.ALMOST_BLACK, plot_utils.ALMOST_BLACK],
                   [plot_utils.ACCENT_COLOR, plot_utils.ACCENT_COLOR],
                   [plot_utils.ACCENT_COLOR, plot_utils.ALMOST_BLACK],
                   [plot_utils.ACCENT_COLOR, plot_utils.ALMOST_BLACK]]
    line_styles = [['-', ':'], ['-', ':'], ['-', '-'], [':', ':']]

    xs = np.array([0., 0.225])
    pad = 0.5
    ax2.text(
      0, 0.9,
      "Percent difference in\npublications per year after\nbecoming a parent",
      fontsize=10, ha='left')
    for i in range(len(markerfacecolors)):
        ys = [0.7-i*.2]*2
        ax2.text(xs[0], ys[0]+.05, labels[i], color=plot_utils.DARK_COLOR,
                 fontsize=9)
        ax2.plot(xs, ys, color=line_colors[i][0], linestyle=line_styles[i][0])
        ax2.plot([np.mean(xs)], [np.mean(ys)], 'o', color=line_colors[i][0],
                 markerfacecolor=markerfacecolors[i][0])
        ax2.text(np.mean([xs[1], pad+xs[0]]), ys[1], 'vs', va='center',
                 ha='center')
        ax2.plot(pad+xs, ys, color=line_colors[i][1], linestyle=line_styles[i][1])
        ax2.plot([pad+np.mean(xs)], [np.mean(ys)], 'o', color=line_colors[i][1],
                 markerfacecolor=markerfacecolors[i][1])
        ax2.text(pad+xs[1], ys[1], ': ', va='center', ha='left')
        ax2.text(2.9*pad+xs[1], ys[1], r'{0:.2f}\%'.format(vals[i]*100),
                 va='center', ha='right')

    # ax2.text(1, -1, "* Men and women without children ", fontsize=8, ha='center')

    plot_utils.finalize(ax)
    ax.set_xticklabels(range(-5, 11), fontsize=9)
    ax.set_yticklabels(range(1, y_max+1), fontsize=9)

    fig.tight_layout()
    plt.savefig('../plots/diff_in_diff/annual_productivity_dd_%s_%s_kid_%s.pdf' %
                (FILE_ENDING, FIELD.lower(), DATE),
                dpi=500)

    women_params = []
    men_params = []
    women_shock_estimates = []
    men_shock_estimates = []
    women_slope_estimates = []
    men_slope_estimates = []

    for iteration in df_m_control['round'].unique():
        men_data = pd.concat([df_m_control[df_m_control['round'] == iteration],
                              df_m_treated])
        # df_m.sample(n=len(df_m_control), replace=True, random_state=iteration)])
        # men_data = men_data[men_data.c > men_data.t]
        women_data = pd.concat([df_w_control[df_w_control['round'] == iteration],
                                df_w_treated])
        # df_w.sample(n=len(df_w_control), replace=True, random_state=iteration)])
        # women_data = women_data[women_data.c > women_data.t]

        formula = 'y ~ t + C(t>0) + t:C(t>0) + C(parent) + C(parent):t + \
                      C(parent):C(t>0) + C(parent):C(t>0):t'
        women_mod = smf.ols(formula=formula, data=women_data)
        women_res = women_mod.fit()

        men_mod = smf.ols(formula=formula, data=men_data)
        men_res = men_mod.fit()

        women_params.append(women_res.params)
        men_params.append(men_res.params)

        women_shock_estimates.append(
          get_params(women_res, 'C(parent)[T.True]:C(t > 0)[T.True]'))
        men_shock_estimates.append(
          get_params(men_res, 'C(parent)[T.True]:C(t > 0)[T.True]'))

        women_slope_estimates.append(
          get_params(women_res, 'C(parent)[T.True]:C(t > 0)[T.True]:t'))
        men_slope_estimates.append(
          get_params(men_res, 'C(parent)[T.True]:C(t > 0)[T.True]:t'))

    def get_linear_fit(params):
        linear_fit_params = {}

        # Pre-treatment intercepts & slopes for non-parents
        linear_fit_params['beta_0'] = np.mean([param['Intercept']
                                              for param in params])
        linear_fit_params['beta_1'] = np.mean([param['t'] for param in params])

        # ... and for parents.
        linear_fit_params['beta_4'] = np.mean([param['C(parent)[T.True]']
                                              for param in params])
        linear_fit_params['beta_5'] = np.mean([param['C(parent)[T.True]:t']
                                              for param in params])

        # Post-treatment intercepts & slopes for non-parents
        linear_fit_params['beta_2'] = np.mean([param['C(t > 0)[T.True]']
                                              for param in params])
        linear_fit_params['beta_3'] = np.mean([param['t:C(t > 0)[T.True]']
                                              for param in params])

        # ... and for parents
        linear_fit_params['beta_6'] = np.mean(
          [param['C(parent)[T.True]:C(t > 0)[T.True]'] for param in params])
        linear_fit_params['beta_7'] = np.mean(
          [param['C(parent)[T.True]:C(t > 0)[T.True]:t'] for param in params])

        return linear_fit_params

    def model(params, t, parent):
        control_pre = params['beta_0'] + params['beta_1']*t
        treated_pre = params['beta_4'] + params['beta_5']*t

        control_post = params['beta_2'] + params['beta_3']*t
        treated_post = params['beta_6'] + params['beta_7']*t

        if not parent:
            if t > 0:
                return control_pre + control_post
            else:
                return control_pre
        else:
            if t > 0:
                return control_pre + treated_pre + control_post + treated_post
            else:
                return control_pre + treated_pre
        return None

    #
    # Plot men's average productivity with and without kids
    #
    fig, ax = plt.subplots(ncols=2, nrows=1,
                           figsize=plot_utils.DOUBLE_FIG_SIZE, sharey=True)

    men_fit = get_linear_fit(men_params)
    trend = np.array([df_m_control[df_m_control['t'] == t]['y'].mean(skipna=True)
                      for t in regression.T])
    ax[0].scatter(regression.T, trend, label='w/o Children', marker='o',
                  color=plot_utils.ALMOST_BLACK, alpha=0.5)
    ax[0].plot(np.arange(-5, 10, 0.1),
               [model(men_fit, t, False) for t in np.arange(-5, 10, 0.1)],
               linestyle='dotted', color=plot_utils.ALMOST_BLACK)

    adjusted_trend = np.array(
      [df_m_treated[df_m_treated['t'] == t]['y'].mean(skipna=True)
       for t in regression.T])
    ax[0].scatter(regression.T, adjusted_trend, label='w/ Children', marker='o',
                  color=plot_utils.ACCENT_COLOR, alpha=0.5)
    ax[0].plot(np.arange(-5, 10, 0.1),
               [model(men_fit, t, True) for t in np.arange(-5, 10, 0.1)],
               color=plot_utils.ACCENT_COLOR)

    # ax[0].axvline(x=0, ls=":")

    # Add text
    ax[0].set_title("Average Annual Productivity (Men)",
                    fontsize=plot_utils.TITLE_SIZE)
    ax[0].set_xlabel("Time Relative to First Child's Birth (Years)")
    ax[0].set_ylabel("Number of Publications")
    ax[0].legend(loc='upper left', fontsize=plot_utils.LEGEND_SIZE,
                 frameon=False, ncol=1)

    ax[0].set_xticks(range(-5, 11))
    # Add styling and save
    plot_utils.finalize(ax[0])

    #
    # Plot women's average productivity with and without kids
    #
    women_fit = get_linear_fit(women_params)
    trend = np.array([df_w_control[df_w_control['t'] == t]['y'].mean(skipna=True)
                      for t in regression.T])
    ax[1].scatter(regression.T, trend, label='w/o Children', marker='o',
                  color=plot_utils.ALMOST_BLACK, alpha=0.5)
    ax[1].plot(np.arange(-5, 10, 0.1),
               [model(women_fit, t, False) for t in np.arange(-5, 10, 0.1)],
               linestyle='dotted', color=plot_utils.ALMOST_BLACK)

    adjusted_trend = np.array(
      [df_w_treated[df_w_treated['t'] == t]['y'].mean(skipna=True)
       for t in regression.T])
    ax[1].scatter(regression.T, adjusted_trend, label='w/ Children', marker='o',
                  color=plot_utils.ACCENT_COLOR, alpha=0.5)
    ax[1].plot(np.arange(-5, 10, 0.1),
               [model(women_fit, t, True) for t in np.arange(-5, 10, 0.1)],
               color=plot_utils.ACCENT_COLOR)

    # Add text
    ax[1].set_title("Average Annual Productivity (Women)",
                    fontsize=plot_utils.TITLE_SIZE)
    ax[1].set_xlabel("Time Relative to First Child's Birth (Years)")

    ax[1].set_xticks(range(-5, 11))
    ax[1].set_ylim(-0.2, y_max+0.2)

    # Add styling and save
    plot_utils.finalize(ax[1])
    plt.tight_layout()
    plt.savefig(
      '../plots/diff_in_diff/annual_productivity_dd_%s_%s_kid_wmodel_%s.pdf' %
      (FILE_ENDING, FIELD.lower(), DATE),
      dpi=500)

    #
    # Plot men's average productivity with and without kids
    #
    fig, ax = plt.subplots(ncols=1, nrows=1,
                           figsize=plot_utils.SINGLE_FIG_SIZE)
    trend = np.array([df_m_control[df_m_control['t'] == t]['y'].mean()
                      for t in regression.T])
    adjusted_trend = np.array([df_m_treated[df_m_treated['t'] == t]['y'].mean()
                              for t in regression.T])

    diffs = [[(df_m_control[(df_m_control['t'] == t)].y.sample(n=1).values[0] - df_m_treated[df_m_treated['t'] == t].y.sample(n=1).values[0])
              for i in range(N_samples*10)] for t in regression.T]
    sems = np.array(
      [np.nanstd(diff)/np.count_nonzero(~np.isnan(diff)) for diff in diffs])

    ax.plot(regression.T, adjusted_trend-trend, linestyle='-', marker='o',
            color=plot_utils.ALMOST_BLACK,
            label='(Fathers - Men w/o Children)')
    ax.fill_between(regression.T, (adjusted_trend-trend)-2*sems,
                    (adjusted_trend-trend)+2*sems,
                    color=plot_utils.ALMOST_BLACK, alpha=0.2)

    #
    # Plot women's average productivity with and without kids
    #
    trend = np.array(
      [df_w_control[df_w_control['t'] == t]['y'].mean() for t in regression.T])
    adjusted_trend = np.array(
      [df_w_treated[df_w_treated['t'] == t]['y'].mean() for t in regression.T])

    diffs = [[(df_w_control[(df_w_control['t'] == t)].y.sample(n=1).values[0] - df_w_treated[df_w_treated['t'] == t].y.sample(n=1).values[0])
              for i in range(N_samples*10)] for t in regression.T]
    sems = np.array(
      [np.nanstd(diff)/np.count_nonzero(~np.isnan(diff)) for diff in diffs])

    ax.plot(regression.T, adjusted_trend-trend, linestyle='-', marker='o',
            color=plot_utils.ACCENT_COLOR,
            label='(Mothers - Women w/o Children)')
    ax.fill_between(regression.T, (adjusted_trend-trend)-2*sems,
                    (adjusted_trend-trend)+2*sems,
                    color=plot_utils.ACCENT_COLOR, alpha=0.2)

    # Move left y-axis and bottim x-axis to centre, passing through (0,0)
    ax.spines['bottom'].set_position('center')

    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # # Show ticks in the left and lower axes only
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # Add text
    ax.set_title("Yearly Difference in Publications",
                 fontsize=plot_utils.TITLE_SIZE)
    ax.set_xlabel("Time Relative to First Child's Birth (Years)")
    ax.set_ylabel("Number of Publications")

    ax.set_xticks(range(-5, 11, 5))
    ax.set_ylim(-2, 2)
    ax.set_yticks(range(-2, 3, 1))
    plt.legend(frameon=False, loc='lower right',
               fontsize=plot_utils.LEGEND_SIZE)
    # Add styling and save
    plot_utils.finalize(ax)
    plt.tight_layout()
    plt.savefig(
      '../plots/diff_in_diff/diff_annual_productivity_dd_%s_%s_kid_%s.pdf' %
      (FILE_ENDING, FIELD.lower(), DATE), dpi=1000)

    print("Relative to assistant professor role")

    #
    # Plot men's average productivity with and without kids
    #
    # fig, ax = plt.subplots(ncols=2, nrows=1,
    #                        figsize=plot_utils.DOUBLE_FIG_SIZE, sharey=True)

    # trend = np.array(
    #   [df_m_control[df_m_control['c'] == t]['y'].mean(skipna=True)
    #    for t in regression.T])
    # std = regression.get_bootstrap_trajectories(df_m_control, N_samples,
    #                                             cumulative=False)

    # ax[0].plot(regression.T, trend, label='w/o Children', linestyle='dotted',
    #            marker='o', markerfacecolor='white',
    #            color=plot_utils.ALMOST_BLACK)
    # ax[0].fill_between(regression.T, trend-2*std, trend+2*std,
    #                    color=plot_utils.ALMOST_BLACK, alpha=0.2)

    # adjusted_trend = np.array(
    #   [df_m_treated[df_m_treated['c'] == t]['y'].mean(skipna=True)
    #    for t in regression.T])
    # std = regression.get_bootstrap_trajectories(df_m_treated, N_samples,
    #                                             cumulative=False)

    # ax[0].plot(regression.T, adjusted_trend, label='w/ Children', linestyle='-',
    #            marker='o', color=plot_utils.ALMOST_BLACK)
    # ax[0].fill_between(regression.T, adjusted_trend-2*std,
    #                    adjusted_trend+2*std, color=plot_utils.ALMOST_BLACK,
    #                    alpha=0.2)
    # ax[0].axvline(x=0, ls=":")

    # # Add text
    # ax[0].set_title("Average Annual Productivity (Men)",
    #                 fontsize=plot_utils.TITLE_SIZE)
    # ax[0].set_xlabel("Years Relative to Assistant Professor")
    # ax[0].set_ylabel("Number of Publications")
    # ax[0].legend(loc='upper left', fontsize=plot_utils.LEGEND_SIZE,
    #              frameon=False, ncol=1)

    # ax[0].set_xticks(range(-5, 11))
    # # Add styling and save
    # plot_utils.finalize(ax[0])

    # #
    # # Plot women's average productivity with and without kids
    # #
    # trend = np.array(
    #   [df_w_control[df_w_control['c'] == t]['y'].mean(skipna=True)
    #    for t in regression.T])
    # std = regression.get_bootstrap_trajectories(df_w_control, N_samples,
    #                                             cumulative=False)

    # ax[1].plot(regression.T, trend, label='w/o Children', linestyle='dotted',
    #            marker='o', markerfacecolor='white', color=plot_utils.ACCENT_COLOR)
    # ax[1].fill_between(regression.T, trend-2*std, trend+2*std,
    #                    color=plot_utils.ACCENT_COLOR, alpha=0.2)

    # adjusted_trend = np.array(
    #   [df_w_treated[df_w_treated['c'] == t]['y'].mean(skipna=True)
    #    for t in regression.T])
    # std = regression.get_bootstrap_trajectories(df_w_treated, N_samples,
    #                                             cumulative=False)

    # ax[1].plot(regression.T, adjusted_trend, label='w/ Children', linestyle='-',
    #            marker='o', color=plot_utils.ACCENT_COLOR)
    # ax[1].fill_between(regression.T, adjusted_trend-2*std, adjusted_trend+2*std,
    #                    color=plot_utils.ACCENT_COLOR, alpha=0.2)
    # ax[1].axvline(x=0, ls=":")
    # # Add text
    # ax[1].set_title("Average Annual Productivity (Women)",
    #                 fontsize=plot_utils.TITLE_SIZE)
    # ax[1].set_xlabel("Years Relative to Assistant Professor")

    # ax[1].set_xticks(range(-5, 11))
    # ax[1].set_ylim(-0.2, y_max+0.2)

    # # Add styling and save
    # plot_utils.finalize(ax[1])
    # plt.tight_layout()
    # plt.savefig(
    #   '../plots/diff_in_diff/annual_productivity_dd_%s_%s_career_%s.pdf' %
    #   (FILE_ENDING, FIELD.lower(), DATE),
    #   dpi=500)

    #
    # Plot men's average productivity with and without kids
    #
    # fig, ax = plt.subplots(ncols=2, nrows=1, figsize=plot_utils.DOUBLE_FIG_SIZE,
    #                        sharey=True)

    # adjusted_trend = np.array(
    #   [np.sum(df_m_treated[df_m_treated['t'] == 0]['c'] == t)
    #    for t in regression.T])
    # adjusted_trend = adjusted_trend/len(df_m_treated[df_m_treated['t'] == 0]['c'])
    # print(adjusted_trend)

    # ax[0].bar(regression.T, adjusted_trend, color=plot_utils.ALMOST_BLACK)

    # ax[0].axvline(x=0, ls=":")
    # # Add text
    # ax[0].set_title("Children Relative to Career",
    #                 fontsize=plot_utils.TITLE_SIZE)
    # ax[0].set_xlabel("Years Relative to Assistant Professor")
    # ax[0].set_ylabel("Proportion of Parents\nHaving Child in Year")

    # ax[0].set_xticks(range(-5, 11))
    # # Add styling and save
    # plot_utils.finalize(ax[0])

    # #
    # # Plot women's average productivity with and without kids
    # #

    # adjusted_trend = np.array(
    #   [np.sum(df_w_treated[df_w_treated['t'] == 0]['c'] == t)
    #    for t in regression.T])
    # adjusted_trend = adjusted_trend/len(df_w_treated[df_w_treated['t'] == 0]['c'])
    # print(adjusted_trend)

    # ax[1].bar(regression.T, adjusted_trend, color=plot_utils.ACCENT_COLOR)

    # ax[1].axvline(x=0, ls=":")
    # # Add text
    # ax[1].set_title("Children Relative to Career",
    #                 fontsize=plot_utils.TITLE_SIZE)
    # ax[1].set_xlabel("Years Relative to Assistant Professor")

    # ax[1].set_xticks(range(-5, 11))
    # # ax[1].set_ylim(-0.01, 0.21)

    # # Add styling and save
    # plot_utils.finalize(ax[1])
    # plt.tight_layout()
    # plt.savefig('../plots/diff_in_diff/kids_%s_career_%s.pdf' %
    #             (FIELD.lower(), DATE), dpi=500)

    print("Effects within control groups")

    #
    # Plot men's average productivity with and without kids
    #
    fig, ax = plt.subplots(ncols=1, nrows=1,
                           figsize=plot_utils.SINGLE_FIG_SIZE, sharey=True)
    trend = cohort_utils.generate_average_cumulative(df_m_control).tolist()
    std = regression.get_bootstrap_trajectories(df_m_control, N_samples)
    ax.plot(regression.T, trend, label='Men', linestyle='-', marker='o',
            color=plot_utils.ALMOST_BLACK)
    ax.fill_between(regression.T, trend-2*std, trend+2*std,
                    color=plot_utils.ALMOST_BLACK, alpha=0.2)

    adjusted_trend = cohort_utils.generate_average_cumulative(df_w_control).tolist()
    std = regression.get_bootstrap_trajectories(df_w_control, N_samples)
    ax.plot(regression.T, adjusted_trend, label='Women', linestyle='-',
            marker='o', color=plot_utils.ACCENT_COLOR)
    ax.fill_between(regression.T, adjusted_trend-2*std, adjusted_trend+2*std,
                    color=plot_utils.ACCENT_COLOR, alpha=0.2)
    ax.axvline(x=0, ls=":")

    # Add text
    ax.set_title("Average Cumulative Productivity (Not-Parents)",
                 fontsize=plot_utils.TITLE_SIZE)
    ax.set_xlabel("Time Relative to First Child's Birth (Years)")
    ax.set_ylabel("Number of Publications")
    ax.legend(loc='upper left', fontsize=plot_utils.LEGEND_SIZE, frameon=False,
              ncol=1)

    ax.text(
      0.95, 0.2, r'$\hat{\beta}_{6}$: %.2f [%.2f, %.2f]' %
      tuple(compute_ci(pd.DataFrame(not_parents_shock_estimates).coef).values()),
      ha='right', va='center', transform=ax.transAxes,
      fontsize=plot_utils.LEGEND_SIZE)
    ax.text(
      0.95, 0.1, r'$\hat{\beta}_{7}$: %.2f [%.2f, %.2f]' %
      tuple(compute_ci(pd.DataFrame(not_parents_slope_estimates).coef).values()),
      ha='right', va='center', transform=ax.transAxes,
      fontsize=plot_utils.LEGEND_SIZE)

    ax.set_xticks(range(-5, 11))
    ax.set_ylim(-2, y_max*15)
    # Add styling and save
    plot_utils.finalize(ax)
    plt.tight_layout()
    plt.savefig(
      '../plots/diff_in_diff/productivity_dd_%s_%s_no_kids_%s.pdf' %
      (FILE_ENDING, FIELD.lower(), DATE),
      dpi=1000)

    print("Effect within treated groups")

    #
    # Plot men's average productivity with and without kids
    #
    fig, ax = plt.subplots(ncols=1, nrows=1,
                           figsize=plot_utils.SINGLE_FIG_SIZE, sharey=True)
    trend = np.array(
      [df_m_treated[df_m_treated['t'] == t]['y'].mean() for t in regression.T])
    std = regression.get_bootstrap_trajectories(df_m_treated, N_samples,
                                                cumulative=False)
    ax.plot(regression.T, trend, label='Men', linestyle='-', marker='o',
            color=plot_utils.ALMOST_BLACK)
    ax.fill_between(regression.T, trend-2*std, trend+2*std,
                    color=plot_utils.ALMOST_BLACK, alpha=0.2)

    adjusted_trend = np.array(
      [df_w_treated[df_w_treated['t'] == t]['y'].mean() for t in regression.T])
    std = regression.get_bootstrap_trajectories(df_w_treated, N_samples,
                                                cumulative=False)
    ax.plot(regression.T, adjusted_trend, label='Women', linestyle='-',
            marker='o', color=plot_utils.ACCENT_COLOR)
    ax.fill_between(regression.T, adjusted_trend-2*std, adjusted_trend+2*std,
                    color=plot_utils.ACCENT_COLOR, alpha=0.2)
    ax.axvline(x=0, ls=":")

    # Add text
    ax.set_title("Average Productivity (Parents)",
                 fontsize=plot_utils.TITLE_SIZE)
    ax.set_xlabel("Time Relative to First Child's Birth (Years)")
    ax.set_ylabel("Number of Publications")
    ax.legend(loc='upper left', fontsize=plot_utils.LEGEND_SIZE, frameon=False,
              ncol=1)

    ax.set_xticks(range(-5, 11))
    # ax.set_ylim(-2, 82)
    # Add styling and save
    plot_utils.finalize(ax)
    plt.tight_layout()
    plt.savefig('../plots/diff_in_diff/productivity_dd_%s_%s_parents_%s.pdf' %
                (FILE_ENDING, FIELD.lower(), DATE),
                dpi=1000)

    # Separate Analysis by Career Stage
    # sum(df_m_treated.c < df_m_treated.t), sum(df_m_control.c < df_m_control.t)

    # fig, ax = plt.subplots(ncols=2, nrows=1,
    #                        figsize=plot_utils.DOUBLE_FIG_SIZE, sharey=True)

    # all_control = pd.concat(
    #   [df_m_control[df_m_control.c <= df_m_control.t],
    #    df_w_control[df_w_control.c <= df_w_control.t]], sort=False)
    # all_treated = pd.concat(
    #   [df_m_treated[df_m_treated.c <= df_m_treated.t],
    #    df_w_treated[df_w_treated.c <= df_w_treated.t]], sort=False)

    # productivity_c = np.array(
    #   [all_control[all_control['t'] == t]['y'].mean() for t in regression.T[:11]])
    # productivity_t = np.array(
    #   [all_treated[all_treated['t'] == t]['y'].mean() for t in regression.T[:11]])

    # ax[0].plot(regression.T[:11], productivity_t, label='Treated',
    #            linestyle='-', marker='o', color=plot_utils.ACCENT_COLOR)
    # # ax.fill_between(T, productivity_w-2*std_w, productivity_w+2*std_w,
    # #                 color='#49d2de', alpha=0.2)
    # ax[0].plot(regression.T[:11], productivity_c, label='Control',
    #            linestyle='-', marker='o', color=plot_utils.ALMOST_BLACK)
    # # ax.fill_between(T, productivity_m-2*std_m, productivity_m+2*std_m,
    # #                 color=plot_utils.ALMOST_BLACK, alpha=0.2)
    # ax[0].axvline(x=0, ls=":")
    # # Add text
    # ax[0].set_title("Cumulative Productivity (before TT start)",
    #                 fontsize=plot_utils.TITLE_SIZE)
    # ax[0].set_ylabel("Number of Publications", fontsize=plot_utils.TITLE_SIZE)
    # ax[0].set_xlabel("Time Relative to First Child's Birth (Years)")
    # ax[0].legend(loc='upper left', fontsize=plot_utils.LEGEND_SIZE,
    #              frameon=False, ncol=1)

    # formula = 'y ~ t:C(parent) + 1'
    # mod = smf.ols(formula=formula, data=pd.concat([all_control, all_treated]))
    # res = mod.fit()

    # control_label = 't:C(parent)[False]'
    # ax[0].text(
    #   0.90, 0.875,
    #   r'slope: %.2f [%.2f, %.2f]' %
    #   (res.params[control_label],
    #    res.conf_int()[0][control_label],
    #    res.conf_int()[1][control_label]),
    #   color=plot_utils.ALMOST_BLACK, ha='right', va='center',
    #   transform=ax[0].transAxes, fontsize=plot_utils.LEGEND_SIZE)
    # treated_label = 't:C(parent)[True]'
    # ax[0].text(
    #   0.90, 0.800,
    #   r'slope: %.2f [%.2f, %.2f]' %
    #   (res.params[treated_label],
    #    res.conf_int()[0][treated_label],
    #    res.conf_int()[1][treated_label]),
    #   color=plot_utils.ACCENT_COLOR, ha='right', va='center',
    #   transform=ax[0].transAxes, fontsize=plot_utils.LEGEND_SIZE)
    # ax[0].set_xlim(-6, 6)
    # ax[0].set_xticks(range(-5, 6))
    # # Add styling and save
    # plot_utils.finalize(ax[0])

    # all_control = pd.concat(
    #   [df_m_control[df_m_control.c > df_m_control.t],
    #    df_w_control[df_w_control.c > df_w_control.t]], sort=False)
    # all_treated = pd.concat(
    #   [df_m_treated[df_m_treated.c > df_m_treated.t],
    #    df_w_treated[df_w_treated.c > df_w_treated.t]], sort=False)

    # productivity_c = np.array(
    #   [all_control[all_control['t'] == t]['y'].mean() for t in regression.T[:11]])
    # productivity_t = np.array(
    #   [all_treated[all_treated['t'] == t]['y'].mean() for t in regression.T[:11]])

    # ax[1].plot(regression.T[:11], productivity_t, label='Treated',
    #            linestyle='-', marker='o', color=plot_utils.ACCENT_COLOR)
    # # ax.fill_between(T, productivity_w-2*std_w, productivity_w+2*std_w,
    # #                 color='#49d2de', alpha=0.2)
    # ax[1].plot(regression.T[:11], productivity_c, label='Control',
    #            linestyle='-', marker='o', color=plot_utils.ALMOST_BLACK)
    # # ax.fill_between(T, productivity_m-2*std_m, productivity_m+2*std_m,
    # #                 color=plot_utils.ALMOST_BLACK, alpha=0.2)
    # ax[1].axvline(x=0, ls=":")
    # # Add text
    # ax[1].set_title("Cumulative Productivity (after TT start)",
    #                 fontsize=plot_utils.TITLE_SIZE)
    # ax[1].set_ylabel("Number of Publications", fontsize=plot_utils.TITLE_SIZE)
    # ax[1].set_xlabel("Time Relative to First Child's Birth (Years)")
    # ax[1].legend(loc='upper left', fontsize=plot_utils.LEGEND_SIZE,
    #              frameon=False, ncol=1)

    # formula = 'y ~ t:C(parent) + 1'
    # mod = smf.ols(formula=formula, data=pd.concat([all_control, all_treated]))
    # res = mod.fit()

    # control_label = 't:C(parent)[False]'
    # ax[1].text(
    #   0.90, 0.275,
    #   r'slope: %.2f [%.2f, %.2f]' %
    #   (res.params[control_label],
    #    res.conf_int()[0][control_label],
    #    res.conf_int()[1][control_label]),
    #   color=plot_utils.ALMOST_BLACK, ha='right', va='center',
    #   transform=ax[1].transAxes, fontsize=plot_utils.LEGEND_SIZE)
    # treated_label = 't:C(parent)[True]'
    # ax[1].text(
    #   0.90, 0.200,
    #   r'slope: %.2f [%.2f, %.2f]' %
    #   (res.params[treated_label],
    #    res.conf_int()[0][treated_label],
    #    res.conf_int()[1][treated_label]),
    #   color=plot_utils.ACCENT_COLOR, ha='right', va='center',
    #   transform=ax[1].transAxes, fontsize=plot_utils.LEGEND_SIZE)
    # ax[1].set_xlim(-6, 6)
    # ax[1].set_xticks(range(-5, 6))
    # # Add styling and save
    # plot_utils.finalize(ax[1])

    # plt.tight_layout()
    # plt.savefig(
    #   '../plots/descriptive/productivity_treat_versus_control_%s_%s.pdf' %
    #   (FILE_ENDING, FIELD.lower()), dpi=1000)
