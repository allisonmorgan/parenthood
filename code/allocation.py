
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
import scipy.stats
import argparse
import csv

from scipy.stats import mannwhitneyu, ks_2samp, chi2_contingency, ttest_ind
from statsmodels.stats.proportion import proportions_ztest
from scripts import plot_utils, regression, cohort_utils, load_data

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Helvetica']

N_samples = 10
iterations = 5000
MIDPOINT = 2000

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
pd.options.mode.chained_assignment = None

# This script accepts a field and date as a command line argument
parser = argparse.ArgumentParser()
parser.add_argument("field", type=str, help="Field you want to consider")
parser.add_argument("date", type=str, help="Date to append to frame file")
parser.add_argument("allocate", type=str, help="Construct control group?")

args = parser.parse_args()
print(args.field, args.date)
FIELD = args.field
DATE = args.date
ALLOCATE = args.allocate

if FIELD in ["History", "Business"]:
    FILE_ENDING = 'raw'
    ADJUSTED = False
elif FIELD == 'CS':
    FILE_ENDING = 'adj'
    ADJUSTED = True
else:
    print("Incorrect field supplied.")
    exit()

color_mapping = {
    'CS': ('#5777D9', '#293866'),
    'Business': ('#CC3A35', '#661D1B'),
    'History': ('#8BCC60', '#466630')
}

plot_utils.ACCENT_COLOR, plot_utils.ALMOST_BLACK = color_mapping[FIELD]

# Load field relevant data
data = load_data.load_all_faculty()
if FIELD == 'CS':
    subset = data[data['likely_department'] == 'Computer Science']
else:
    subset = data[data['likely_department'] == FIELD]

subset.loc[:, 'age (actual)'] = pd.Series(2017 - subset['age_coded'])
subset.rename(
    columns={
        'gender_ans': 'gender', 'prestige_inv': 'prestige_frame',
        'prestige_rank_inv': 'prestige_rank',
        'parleave_objective_length_women_inv':
        'parleave_objective_length_women',
        'parleave_objective_type_women_inv': 'parleave_objective_type_women',
        'parleave_objective_length_men_inv': 'parleave_objective_length_men',
        'parleave_objective_type_men_inv': 'parleave_objective_type_men'
    },
    inplace=True)
subset_cols = [
        'firstname', 'lastname', 'name', 'age_coded', 'gender',
        'age (actual)', 'chage1', 'chage2', 'chage3', 'chage4', 'chage5',
        'chage6', 'chage7', 'chage8', 'p1_gender', 'p2_gender', 'p1_edu',
        'p2_edu', 'p1_empl', 'p2_empl', 'curtitle', 'curtitle_other',
        'first_asst_job_year', 'dblp_pubs', 'workhours', 'service_c',
        'service_nc', 'parstigma1', 'parstigma2', 'white', 'hisp', 'black',
        'asian', 'native', 'hawaii', 'otherace', 'narace', 'first_child_birth',
        'children_no', 'prestige_frame', 'prestige_rank',
        'parleave_objective_length_women', 'parleave_objective_type_women',
        'parleave_objective_length_men', 'parleave_objective_type_men',
        'children', 'university_name_standard', 'parleave_elig_child1',
        'parleave_elig_child2', 'parleave_elig_child3',
        'parleave_elig_child4', 'parleave_taken_child1',
        'parleave_taken_child2', 'parleave_taken_child3',
        'parleave_taken_child4', 'isyoung', 'parleave_ideal',
        'current_parleave', 'pid', 'sid']
subset = subset[subset_cols].copy(deep=True)

print(subset[~subset.dblp_pubs.isnull()].shape, subset.shape)

print("Completed merging frame and responses!")

print("Gender representation in our sample.")
women = subset[subset['gender'] == 1].copy(deep=True)
print(women.shape, sum(women['dblp_pubs'].isna())/len(women.dblp_pubs))
men = subset[subset['gender'] == 2].copy(deep=True)
print(men.shape, sum(men['dblp_pubs'].isna())/len(men.dblp_pubs))

print("What fraction of our survey respondents who are age 40+, report having \
      no kids?")
print('Total:\t', subset[subset['age (actual)'] >= 40].children.value_counts(
        normalize=True))
print('Men:\t',
      subset[(subset['age (actual)'] >= 40) &
             (subset['gender'] == 2.0)].children.value_counts(normalize=True))
print('Women:\t',
      subset[(subset['age (actual)'] >= 40) &
             (subset['gender'] == 1.0)].children.value_counts(normalize=True))


count = np.array([sum(subset[(subset['age (actual)'] >= 40) &
                             (subset['gender'] == 1.0)].children == 1),
                  sum(subset[(subset['age (actual)'] >= 40) &
                             (subset['gender'] == 2.0)].children == 1)],
                 dtype=float)
nobs = np.array([sum(subset[(subset['age (actual)'] >= 40) &
                            (subset['gender'] == 1.0)].children.isin([1, 2])),
                 sum(subset[(subset['age (actual)'] >= 40) &
                            (subset['gender'] == 2.0)].children.isin([1, 2]))],
                dtype=float)
print('Z-test:\t', proportions_ztest(count, nobs))


had_kids = subset[subset['children_no'] > 0].copy(deep=True)
had_kids['children_no'].hist(density=True, color=plot_utils.ALMOST_BLACK)

print("Among those that had kids -- men & women -- what were the median number \
      of kids?")
print('Men:\t', had_kids[had_kids.gender == 2]['children_no'].median(),
      '\tWomen:\t', had_kids[had_kids.gender == 1]['children_no'].median())


# Statistical tests between the means & medians
print("KS-test:\t", ks_2samp(had_kids[had_kids.gender == 2]['children_no'],
                             had_kids[had_kids.gender == 1]['children_no']))
print("Mann Whitney test:\t",
      mannwhitneyu(had_kids[had_kids.gender == 2]['children_no'],
                   had_kids[had_kids.gender == 1]['children_no']))

# Statistical tests between the categorical distributions
men_kid_dist = [(had_kids[had_kids.gender == 2]['children_no'] == i).sum()
                for i in range(1, 7)]
women_kid_dist = [(had_kids[had_kids.gender == 1]['children_no'] == i).sum()
                  for i in range(1, 7)]

print("Chi^2:\t", chi2_contingency([women_kid_dist, men_kid_dist],
                                   correction=False))

# Among everyone
men_kid_dist = [(subset[subset.gender == 2]['children_no'] == i).sum()
                for i in range(1, 7)]
women_kid_dist = [(subset[subset.gender == 1]['children_no'] == i).sum()
                  for i in range(1, 7)]

print(chi2_contingency([women_kid_dist, men_kid_dist], correction=False))

# Essentially, there exists statistically significant differences in the means,
# medians, and distributions of children for men and women.

# Basic information about number of men and women & publication data
print("Number of observations with publication data: ",
      sum(subset['dblp_pubs'].isna()), len(subset.dblp_pubs))

women = subset[subset['gender'] == 1].copy(deep=True)
men = subset[subset['gender'] == 2].copy(deep=True)
print("Number of women:\t", women.shape, "\tNumber of men:\t", men.shape)

# ### Plots of productivity relative to career age

# These functions generate productivity trajectories organized relative to
# tenure-track start date. Plot men and women's average cumulative productivity
#
custom_figsize = (4, 4)
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=custom_figsize, sharex=True,
                       sharey=True)
kwargs = {'linestyle': '-', 'marker': 'o', 'fillstyle': 'left'}
(pop_name, female_pop, male_pop, ) = ('Total', women, men)
df_w_adj_pubs = cohort_utils.compute_publication_trend(
    female_pop, -5, 10, adjusted=ADJUSTED, relative_to='first_asst_job_year')
df_m_adj_pubs = cohort_utils.compute_publication_trend(
    male_pop, -5, 10, adjusted=ADJUSTED, relative_to='first_asst_job_year')
print(df_w_adj_pubs.shape, df_m_adj_pubs.shape)

df_w_adj_pubs.loc[:, 'cumulative'] = df_w_adj_pubs.groupby(['i', 'round'])['y'].cumsum()
df_m_adj_pubs.loc[:, 'cumulative'] = df_m_adj_pubs.groupby(['i', 'round'])['y'].cumsum()

productivity_m = df_m_adj_pubs.groupby(['t'])['cumulative'].mean()
productivity_w = df_w_adj_pubs.groupby(['t'])['cumulative'].mean()

print(productivity_m, productivity_w)
std_m = regression.get_bootstrap_trajectories(df_m_adj_pubs, N_samples)
std_w = regression.get_bootstrap_trajectories(df_w_adj_pubs, N_samples)

N = len(female_pop) + len(male_pop)
NM_wpubs = len(df_m_adj_pubs['i'].unique())
NF_wpubs = len(df_w_adj_pubs['i'].unique())
print(pop_name, N, NM_wpubs+NF_wpubs)

# Plot trends
ax.plot(regression.T, productivity_w, label='Female',
        color=plot_utils.ACCENT_COLOR, zorder=5, **kwargs)
ax.fill_between(regression.T, (productivity_w-2*std_w),
                (productivity_w+2*std_w), color=plot_utils.ACCENT_COLOR,
                alpha=0.4, zorder=2)

ax.plot(regression.T, productivity_m, label='Male',
        color=plot_utils.ALMOST_BLACK, zorder=5, **kwargs)
ax.fill_between(regression.T, (productivity_m-2*std_m),
                (productivity_m+2*std_m), color=plot_utils.ALMOST_BLACK,
                alpha=0.4, zorder=2)

ax.set_xlim(-6, 11)
ax.set_xticks(range(-5, 11, 5))

ax.annotate('A', xy=(-10, 5), fontsize=plot_utils.TITLE_SIZE, weight='bold',
            color='0.125')
ax.annotate('%d' % (round(productivity_m[10] - productivity_w[10])),
            xy=(10, productivity_w[10]),
            xytext=(12, productivity_w[10]+0.5*(productivity_m[10]-productivity_w[10])),
            arrowprops=dict(arrowstyle="-", connectionstyle="arc3",
                            color='0.125'),
            color=plot_utils.LIGHT_COLOR)
ax.annotate('%d' % round((productivity_m[10] - productivity_w[10])),
            xy=(10, productivity_m[10]),
            xytext=(12, productivity_w[10]+0.5*(productivity_m[10]-productivity_w[10])),
            arrowprops=dict(arrowstyle="-", connectionstyle="arc3",
                            color='0.125'),
            color='0.125')
print('Long-run W/M (Total)', (productivity_w[10]/productivity_m[10]),
      productivity_m[10]-productivity_w[10])

ins = ax.inset_axes([0.15, 0.5, 0.3, 0.5])
kwargs = {'linestyle': 'dotted', 'marker': 'o', 'markerfacecolor': 'white',
          'markersize': 2, 'linewidth': 1}
pop_name = 'Not Parents'
(female_pop, male_pop) = (women[women.chage1.isna()], men[men.chage1.isna()])
df_w_adj_pubs = cohort_utils.compute_publication_trend(
    female_pop, -5, 10, adjusted=ADJUSTED, relative_to='first_asst_job_year')
df_m_adj_pubs = cohort_utils.compute_publication_trend(
    male_pop, -5, 10, adjusted=ADJUSTED, relative_to='first_asst_job_year')

df_w_adj_pubs.loc[:, 'cumulative'] = df_w_adj_pubs.groupby(['i', 'round'])['y'].cumsum()
df_m_adj_pubs.loc[:, 'cumulative'] = df_m_adj_pubs.groupby(['i', 'round'])['y'].cumsum()

productivity_m = cohort_utils.generate_average_cumulative(df_m_adj_pubs)
productivity_w = cohort_utils.generate_average_cumulative(df_w_adj_pubs)

std_m = regression.get_bootstrap_trajectories(df_m_adj_pubs, N_samples)
std_w = regression.get_bootstrap_trajectories(df_w_adj_pubs, N_samples)

N = len(female_pop) + len(male_pop)
NM_wpubs = len(df_m_adj_pubs['i'].unique())
NF_wpubs = len(df_w_adj_pubs['i'].unique())
print(pop_name, N, NM_wpubs+NF_wpubs)

# Plot trends
ins.plot(regression.T, productivity_w, label='Female',
         color=plot_utils.ACCENT_COLOR, zorder=5, **kwargs)
ins.fill_between(regression.T, (productivity_w-2*std_w),
                 (productivity_w+2*std_w), color=plot_utils.ACCENT_COLOR,
                 alpha=0.4, zorder=2)

ins.plot(regression.T, productivity_m, label='Male',
         color=plot_utils.ALMOST_BLACK, zorder=5, **kwargs)
ins.fill_between(regression.T, (productivity_m-2*std_m),
                 (productivity_m+2*std_m), color=plot_utils.ALMOST_BLACK,
                 alpha=0.4, zorder=2)

ins.set_xlim(-6, 11)
ins.set_xticks(range(-5, 11, 5))

ins.annotate('%d' % (round(productivity_m[10] - productivity_w[10])),
             xy=(10, productivity_w[10]),
             xytext=(12, productivity_w[10]+0.5*(productivity_m[10]-productivity_w[10])),
             arrowprops=dict(arrowstyle="-", connectionstyle="arc3",
                             color='0.125'),
             color=plot_utils.LIGHT_COLOR)
ins.annotate('%d' % (round(productivity_m[10] - productivity_w[10])),
             xy=(10, productivity_m[10]),
             xytext=(12, productivity_w[10]+0.5*(productivity_m[10]-productivity_w[10])),
             arrowprops=dict(arrowstyle="-", connectionstyle="arc3",
                             color='0.125'),
             color='0.125')
print('Long-run W/M (Not Parents)', (productivity_w[10]/productivity_m[10]),
      productivity_m[10]-productivity_w[10])

# ins.set_ylim(-5.5, 45.5)
plot_utils.finalize(ins)
# ins.annotate('Not\nParents', xy=(10, -4.5), xytext=(10, -4.5),
#              color=plot_utils.DARK_COLOR,
#              fontsize=plot_utils.LEGEND_SIZE-2, ha='right')
ins.tick_params(labelsize=9)


# ax.set_ylim(-5.5, 45.5)
plot_utils.finalize(ax)

# ax.annotate('Men', xy=(7, 28.), xytext=(7, 28.),
#             color=plot_utils.ALMOST_BLACK,
#             fontsize=plot_utils.LEGEND_SIZE)
# ax.annotate('Women', xy=(7.5, 11.5), xytext=(7.5, 11.5),
#             color=plot_utils.ACCENT_COLOR,
#             fontsize=plot_utils.LEGEND_SIZE)
# ax.annotate('Total', xy=(10, -4.5), xytext=(10, -4.5),
#             color=plot_utils.DARK_COLOR,
#             fontsize=plot_utils.LEGEND_SIZE, ha='right')

ax.set_ylabel("Number of Publications", fontsize=plot_utils.TITLE_SIZE)
ax.set_xlabel("Time Relative to\nAssistant Professor (Years)",
              fontsize=plot_utils.TITLE_SIZE)
plt.tight_layout()
plt.savefig(
    '../plots/descriptive/small_career_productivity_%s.pdf' % FIELD.lower(),
    dpi=1000, figsize=custom_figsize)

# Plot fathers and mothers' average cumulative productivity
custom_figsize = (4, 4)
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=custom_figsize, sharex=True,
                       sharey=True)

kwargs = {'linestyle': '-', 'marker': 'o', 'fillstyle': 'left'}
pop_name = 'Parent'
(female_pop, male_pop) = (women[~women.chage1.isna()], men[~men.chage1.isna()])
df_w_adj_pubs = cohort_utils.compute_publication_trend(
    female_pop, -5, 10, adjusted=ADJUSTED, relative_to='first_asst_job_year')
df_m_adj_pubs = cohort_utils.compute_publication_trend(
    male_pop, -5, 10, adjusted=ADJUSTED, relative_to='first_asst_job_year')

df_w_adj_pubs.loc[:, 'cumulative'] = df_w_adj_pubs.groupby(['i', 'round'])['y'].cumsum()
df_m_adj_pubs.loc[:, 'cumulative'] = df_m_adj_pubs.groupby(['i', 'round'])['y'].cumsum()

productivity_m = cohort_utils.generate_average_cumulative(df_m_adj_pubs)
productivity_w = cohort_utils.generate_average_cumulative(df_w_adj_pubs)

N_samples = 10
std_m = regression.get_bootstrap_trajectories(df_m_adj_pubs, N_samples)
std_w = regression.get_bootstrap_trajectories(df_w_adj_pubs, N_samples)

N = len(female_pop) + len(male_pop)
NM_wpubs = len(df_m_adj_pubs['i'].unique())
NF_wpubs = len(df_w_adj_pubs['i'].unique())
print(pop_name, N, NM_wpubs+NF_wpubs)

# Plot trends
ax.plot(regression.T, productivity_w, label='Female',
        color=plot_utils.ACCENT_COLOR, zorder=5, **kwargs)
ax.fill_between(regression.T, (productivity_w-2*std_w),
                (productivity_w+2*std_w), color=plot_utils.ACCENT_COLOR,
                alpha=0.4, zorder=2)

ax.plot(regression.T, productivity_m, label='Male',
        color=plot_utils.ALMOST_BLACK, zorder=5, **kwargs)
ax.fill_between(regression.T, (productivity_m-2*std_m),
                (productivity_m+2*std_m), color=plot_utils.ALMOST_BLACK,
                alpha=0.4, zorder=2)

ax.set_xlim(-6, 11)
ax.set_xticks(range(-5, 11, 5))

print('Long-run W/M (Parents)', (productivity_w[10]/productivity_m[10]),
      productivity_m[10]-productivity_w[10])
print('Men: ', productivity_m[10], 'Women: ', productivity_w[10])

# Calculate the number of years missing for mothers to reach fathers (relative
# to their child's birth)
df_w_adj_pubs = cohort_utils.compute_publication_trend(
    female_pop, -5, 10, adjusted=ADJUSTED, relative_to='first_child_birth')
df_w_adj_pubs = df_w_adj_pubs[df_w_adj_pubs.t > 0]
df_m_adj_pubs = cohort_utils.compute_publication_trend(
    male_pop, -5, 10, adjusted=ADJUSTED, relative_to='first_child_birth')
df_m_adj_pubs = df_m_adj_pubs[df_m_adj_pubs.t > 0]

df_w_adj_pubs.loc[:, 'cumulative'] = df_w_adj_pubs.groupby(['i', 'round'])['y'].cumsum()
df_m_adj_pubs.loc[:, 'cumulative'] = df_m_adj_pubs.groupby(['i', 'round'])['y'].cumsum()

productivity_m = cohort_utils.generate_average_cumulative(df_m_adj_pubs)
productivity_w = cohort_utils.generate_average_cumulative(df_w_adj_pubs)

print('Lost years?', (productivity_m[10]-productivity_w[10]),
      (productivity_m[10]-productivity_w[10])/(productivity_w[10]/len(productivity_w)),
      len(productivity_w))

# ax.set_ylim(-2.5, 82.5)
plot_utils.finalize(ax)

ax.set_ylabel("Number of Publications", fontsize=plot_utils.TITLE_SIZE)
ax.set_xlabel("Time Relative to\nAssistant Professor (Years)",
              fontsize=plot_utils.TITLE_SIZE)
plt.tight_layout()
plt.savefig(
    '../plots/descriptive/parents_career_productivity_%s.pdf' % FIELD.lower(),
    dpi=1000, figsize=custom_figsize)


# Does the gap grow with the number of children?
kwargs = {'linestyle': '-', 'marker': 'o', 'fillstyle': 'left'}
pop_name = 'Parent'
(female_pop, male_pop) = (women[women.children_no == 1],
                          men[men.children_no == 1])
df_w_adj_pubs = cohort_utils.compute_publication_trend(
    female_pop, -5, 10, adjusted=ADJUSTED, relative_to='first_child_birth')
df_w_adj_pubs = df_w_adj_pubs[df_w_adj_pubs.t > 0]
df_m_adj_pubs = cohort_utils.compute_publication_trend(
    male_pop, -5, 10, adjusted=ADJUSTED, relative_to='first_child_birth')
df_m_adj_pubs = df_m_adj_pubs[df_m_adj_pubs.t > 0]

df_w_adj_pubs.loc[:, 'cumulative'] = df_w_adj_pubs.groupby(['i', 'round'])['y'].cumsum()
df_m_adj_pubs.loc[:, 'cumulative'] = df_m_adj_pubs.groupby(['i', 'round'])['y'].cumsum()

productivity_m = cohort_utils.generate_average_cumulative(df_m_adj_pubs)
productivity_w = cohort_utils.generate_average_cumulative(df_w_adj_pubs)
print('Parents with one child (N = %f)' %
      int(female_pop.shape[0] + male_pop.shape[0]),
      productivity_m[10] - productivity_w[10], productivity_m[10],
      productivity_w[10])

kwargs = {'linestyle': '-', 'marker': 'o', 'fillstyle': 'left'}
pop_name = 'Parent'
(female_pop, male_pop) = (women[women.children_no >= 2],
                          men[men.children_no >= 2])
df_w_adj_pubs = cohort_utils.compute_publication_trend(
    female_pop, -5, 10, adjusted=False, relative_to='first_child_birth')
df_w_adj_pubs = df_w_adj_pubs[df_w_adj_pubs.t > 0]
df_m_adj_pubs = cohort_utils.compute_publication_trend(
    male_pop, -5, 10, adjusted=False, relative_to='first_child_birth')
df_m_adj_pubs = df_m_adj_pubs[df_m_adj_pubs.t > 0]

df_w_adj_pubs.loc[:, 'cumulative'] = df_w_adj_pubs.groupby(['i', 'round'])['y'].cumsum()
df_m_adj_pubs.loc[:, 'cumulative'] = df_m_adj_pubs.groupby(['i', 'round'])['y'].cumsum()

productivity_m = cohort_utils.generate_average_cumulative(df_m_adj_pubs)
productivity_w = cohort_utils.generate_average_cumulative(df_w_adj_pubs)
print('Parents with two or more children (N = %f)' %
      int(female_pop.shape[0] + male_pop.shape[0]),
      productivity_m[10] - productivity_w[10], productivity_m[10],
      productivity_w[10])

# Plot publications over time for this field
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(4, 3))

df_adj_pubs = cohort_utils.compute_publication_trend(
    pd.concat([women, men]), -5, 10, adjusted=ADJUSTED,
    relative_to='first_asst_job_year')
df_adj_pubs.loc[:, 'cumulative'] = df_adj_pubs.groupby(['i', 'round'])['y'].cumsum()

avg_in_first_five = []
tt_first_year = []
for index, i in enumerate(list(df_adj_pubs['i'].unique())):
    individual = df_adj_pubs[(df_adj_pubs['i'] == i)]
    individual_tt_start = individual[individual['t'] == 0]['s'].values[0]
    if individual_tt_start >= 2015:
        continue
    tt_first_year.append(individual_tt_start)
    avg_in_first_five.append(np.nanmean([
        individual[(individual['t'] == 0)]['y'].values[0],
        individual[(individual['t'] == 1)]['y'].values[0],
        individual[(individual['t'] == 2)]['y'].values[0],
        individual[(individual['t'] == 3)]['y'].values[0],
        individual[(individual['t'] == 4)]['y'].values[0]]))

plt.scatter(tt_first_year, avg_in_first_five, color=plot_utils.ACCENT_COLOR,
            alpha=0.2)

mod = smf.ols(formula='pubs ~ t',
              data=pd.DataFrame({'t': tt_first_year,
                                 'pubs': avg_in_first_five}))
res = mod.fit()
print(res.summary())

plt.plot(list(range(1965, 2015, 1)),
         res.predict(pd.DataFrame({'t': list(range(1965, 2015, 1))})),
         '--', linewidth=2, color=plot_utils.ALMOST_BLACK)
# plt.ylim(-0.5, 4)

ax.set_ylabel("Avg. Annual Productivity", fontsize=plot_utils.TITLE_SIZE)
ax.set_xlabel("Assistant Professor Position (Year)",
              fontsize=plot_utils.TITLE_SIZE)
ax.annotate('slope: {%0.3f}' % res.params['t'], xy=(1960, 3.7),
            xytext=(1960, 3.7), color=plot_utils.DARK_COLOR,
            fontsize=plot_utils.LEGEND_SIZE, ha='left')

plot_utils.finalize(ax)
plt.tight_layout()
plt.savefig('../plots/descriptive/pubs_over_time_%s.pdf' % FIELD.lower(),
            dpi=500, figsize=(4, 3))


# Plots of productivity relative to parenthood
# Here we are putting all of this publication and authorship data together for
# modeling outside of this notebook.
df_w = cohort_utils.compute_publication_trend(women, -5, 10, adjusted=ADJUSTED)
assert len(df_w['i'].unique()) > 0

df_m = cohort_utils.compute_publication_trend(men, -5, 10, adjusted=ADJUSTED)
assert len(df_m['i'].unique()) > 0

# Plot men and women's average productivity
productivity_m = np.array([df_m[df_m['t'] == t]['y'].mean()
                           for t in regression.T], dtype=float)
productivity_w = np.array([df_w[df_w['t'] == t]['y'].mean()
                           for t in regression.T], dtype=float)

productivity_trajectory = [None]*N_samples
for i in range(N_samples):
    sample = regression.get_sample(df_w)
    productivity_trajectory[i] = [sample[sample['t'] == t]['y'].mean()
                                  for t in regression.T]
std_w = scipy.stats.sem(productivity_trajectory, axis=0)

productivity_trajectory = [None]*N_samples
for i in range(N_samples):
    sample = regression.get_sample(df_m)
    productivity_trajectory[i] = [sample[sample['t'] == t]['y'].mean()
                                  for t in regression.T]
std_m = scipy.stats.sem(productivity_trajectory, axis=0)

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=plot_utils.SINGLE_FIG_SIZE)

ax.plot(regression.T, productivity_w, label='Mothers', linestyle='-',
        marker='o', color=plot_utils.ACCENT_COLOR, zorder=2)
ax.fill_between(regression.T, productivity_w-2*std_w, productivity_w+2*std_w,
                color=plot_utils.ACCENT_COLOR, alpha=0.4)
ax.plot(regression.T, productivity_m, label='Fathers', linestyle='-',
        marker='o', color=plot_utils.ALMOST_BLACK, zorder=2)
ax.fill_between(regression.T, productivity_m-2*std_m, productivity_m+2*std_m,
                color=plot_utils.ALMOST_BLACK, alpha=0.4)
ax.axvline(x=0, ls=":", zorder=1, color=plot_utils.ALMOST_BLACK)
# Add text
ax.set_title("Average Yearly Productivity", fontsize=plot_utils.TITLE_SIZE)
ax.set_ylabel("Number of Publications", fontsize=plot_utils.TITLE_SIZE)
ax.set_xlabel("Time Relative to First Child's Birth (Years)")
ax.legend(loc='upper left', fontsize=plot_utils.LEGEND_SIZE, frameon=False,
          ncol=1)
ax.text(0.90, 0.075,
        'Long-run Annual Difference: %.2f' %
        (productivity_m[10]-productivity_w[10]),
        ha='right', va='center',
        transform=ax.transAxes, fontsize=plot_utils.LEGEND_SIZE,
        backgroundcolor='white', zorder=3)
ax.set_xlim(-6, 11)
ax.set_xticks(range(-5, 11))
# Add styling and save
plot_utils.finalize(ax)
plt.tight_layout()
plt.savefig('../plots/descriptive/yearly_productivity_%s.pdf' %
            (FIELD.lower()), dpi=1000, figsize=plot_utils.SINGLE_FIG_SIZE)


# Plot men and women's average cumulative productivity
df_m.loc[:, 'cumulative'] = df_m.groupby(['i', 'round'])['y'].cumsum()
df_w.loc[:, 'cumulative'] = df_w.groupby(['i', 'round'])['y'].cumsum()

productivity_m = df_m.groupby(['t'])['cumulative'].mean()
productivity_w = df_w.groupby(['t'])['cumulative'].mean()

N_samples = 10
std_m = regression.get_bootstrap_trajectories(df_m, N_samples)
std_w = regression.get_bootstrap_trajectories(df_w, N_samples)

#
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=plot_utils.SINGLE_FIG_SIZE)

ax.plot(regression.T, productivity_w, label='Mothers', linestyle='-',
        marker='o', color=plot_utils.ACCENT_COLOR, zorder=2)
ax.fill_between(regression.T, productivity_w-2*std_w, productivity_w+2*std_w,
                color=plot_utils.ACCENT_COLOR, alpha=0.4)
ax.plot(regression.T, productivity_m, label='Fathers', linestyle='-',
        marker='o', color=plot_utils.ALMOST_BLACK, zorder=2)
ax.fill_between(regression.T, productivity_m-2*std_m, productivity_m+2*std_m,
                color=plot_utils.ALMOST_BLACK, alpha=0.4)

ax.axvline(x=0, ls=":", zorder=1, color=plot_utils.ALMOST_BLACK)
# Add text
ax.set_title("Average Cumulative Productivity", fontsize=plot_utils.TITLE_SIZE)
ax.set_ylabel("Number of Publications", fontsize=plot_utils.TITLE_SIZE)
ax.set_xlabel("Time Relative to First Child's Birth (Years)")
print('MEN: ', list(productivity_m), '\n')
print('WOMEN: ', list(productivity_w), '\n')
ax.text(0.90, 0.075,
        'Long-run Cumulative Difference: %.2f' %
        (productivity_m[10]-productivity_w[10]),
        ha='right', va='center', transform=ax.transAxes,
        fontsize=plot_utils.LEGEND_SIZE, backgroundcolor='white', zorder=3)
ax.legend(loc='upper left', fontsize=plot_utils.LEGEND_SIZE, frameon=False,
          ncol=1)
ax.set_xlim(-6, 11)
ax.set_xticks(range(-5, 11))
# Add styling and save
plot_utils.finalize(ax)
plt.tight_layout()
plt.savefig(
    '../plots/descriptive/cumulative_productivity_%s.pdf' % (FIELD.lower()),
    dpi=1000, figsize=plot_utils.SINGLE_FIG_SIZE)

#
# Plot men and women's age of first kid
#
parenthood_m = [
    (df_m[df_m['i'] == index]['c'] - df_m[df_m['i'] == index]['t']).values[0]
    for index in np.unique(df_m['i'])]
parenthood_w = [
    (df_w[df_w['i'] == index]['c'] - df_w[df_w['i'] == index]['t']).values[0]
    for index in np.unique(df_w['i'])]
#
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=plot_utils.SINGLE_FIG_SIZE,
                       sharex=True, sharey=True)
ax.hist([parenthood_m, parenthood_w], density=True,
        color=[plot_utils.ALMOST_BLACK, plot_utils.ACCENT_COLOR],
        label=['Men', 'Women'], align='right')
# ax.set_xticks([15, 20, 25, 30, 35, 40, 45, 50, 55])
# ax.set_xticklabels(['', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44',
#                     '45-49', '50-55'], fontsize=plot_utils.LABEL_SIZE,
#                     rotation=45)
ax.legend(loc='upper left', fontsize=plot_utils.LEGEND_SIZE, frameon=False,
          ncol=1)
# Add text
ax.set_ylabel("Percent of Parents", fontsize=plot_utils.TITLE_SIZE)
ax.set_xlabel("Career Age at their First Child's Birth",
              fontsize=plot_utils.TITLE_SIZE)
ax.set_title("Career Age at First Child", fontsize=plot_utils.TITLE_SIZE)
# Add styling and save
plot_utils.finalize(ax)
plt.tight_layout()
plt.savefig('../plots/descriptive/parent_career_age_%s.pdf' % (FIELD.lower()),
            dpi=1000, figsize=plot_utils.SINGLE_FIG_SIZE)
#


# Plot men and women's age of first kid
parenthood_m = [
    (df_m[df_m['i'] == index]['age'] - df_m[df_m['i'] == index]['t']).values[0]
    for index in np.unique(df_m['i'])]
parenthood_w = [
    (df_w[df_w['i'] == index]['age'] - df_w[df_w['i'] == index]['t']).values[0]
    for index in np.unique(df_w['i'])]

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=plot_utils.SINGLE_FIG_SIZE,
                       sharex=True, sharey=True)
ax.hist([parenthood_m, parenthood_w],
        bins=[15, 20, 25, 30, 35, 40, 45, 50, 55],
        density=True, color=[plot_utils.ALMOST_BLACK, plot_utils.ACCENT_COLOR],
        label=['Men', 'Women'], align='right')
# ax.set_xticks([15, 20, 25, 30, 35, 40, 45, 50, 55])
# ax.set_xticklabels(['', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44',
#                     '45-49', '50-55'], fontsize=LABEL_SIZE, rotation=45)
ax.legend(loc='upper right', fontsize=plot_utils.LEGEND_SIZE, frameon=False,
          ncol=1)
# Add text
ax.set_ylabel("Percent of Parents", fontsize=plot_utils.TITLE_SIZE)
ax.set_xlabel("Age of Parent at their First Child's Birth",
              fontsize=plot_utils.TITLE_SIZE)
ax.set_title("Age at First Child", fontsize=plot_utils.TITLE_SIZE)
# Add styling and save
plot_utils.finalize(ax)
plt.tight_layout()
plt.savefig(
    '../plots/descriptive/parent_biological_age_%s.pdf' % (FIELD.lower()),
    dpi=1000, figsize=plot_utils.SINGLE_FIG_SIZE)
#


# Generate the control group.
# There is not a significant difference in the proportions of mothers and
# fathers in our survey sample.
print("Proportions of parents and non-parents?")
non_parents = sum(subset['first_child_birth'].isna())
parents = len(subset['first_child_birth']) - non_parents
print('Number of non-parents: %d, Number of parents: %d (%f)' %
      (non_parents, parents, parents/(parents+non_parents)))

non_mothers = sum(women['first_child_birth'].isna())
mothers = len(women['first_child_birth']) - non_mothers
print('Number of non-mothers: %d, Number of mothers: %d (%f)' %
      (non_mothers, mothers, mothers/(non_mothers+mothers)))

non_fathers = sum(men['first_child_birth'].isna())
fathers = len(men['first_child_birth']) - non_fathers
print('Number of non-fathers: %d, Number of fathers: %d (%f)' %
      (non_fathers, fathers, fathers/(non_fathers+fathers)))

count = np.array([mothers, fathers], dtype=object)
nobs = np.array([mothers+non_mothers, fathers+non_fathers], dtype=object)
print("Z-test:\t", proportions_ztest(count, nobs, alternative='smaller'))

# Again, there is no statistical difference in age at first birth for men or
# women
print("Age at which fathers and mothers become parents?")
parents = subset[subset['children_no'] > 0]
parents = parents[parents['gender'].isin([1, 2])]
print(len(parents), len(parents[parents.gender == 1]),
      len(parents[parents.gender == 2]))

print(np.nanmean(parents['chage1'] - (2017 - parents['age (actual)'])))
print(np.nanmean(parents[parents.gender == 1]['chage1'] -
                 (2017.0-parents[parents.gender == 1]['age (actual)'])))
print(np.nanmean(parents[parents.gender == 2]['chage1'] -
                 (2017.0-parents[parents.gender == 2]['age (actual)'])))

print(ttest_ind(
    (parents[parents.gender == 1]['chage1'] -
     (2017.0-parents[parents.gender == 1]['age (actual)'])).dropna(),
    (parents[parents.gender == 2]['chage1'] -
     (2017.0-parents[parents.gender == 2]['age (actual)'])).dropna()))


# No strong correlation between the age at which you have your kid and the
# prestige of your current institution.
print("Age of parenthood and prestige of current institution?")
print(parents.shape,
      parents[parents['chage1'] < parents['first_asst_job_year']].shape,
      parents[parents['chage1'] >= parents['first_asst_job_year']].shape)

parents['prestige_frame'] = pd.to_numeric(parents.prestige_frame)

formula = 'age ~ prestige_rank'

data = subset[subset['children_no'] > 0]
data = data[data['gender'].isin([1, 2])]
data = data[~data.prestige_rank.isnull()]
data['age'] = data['chage1'] - (2017.0 - data['age (actual)'])

mod = smf.ols(formula=formula, data=data)
res = mod.fit()
print(res.summary())

print("Average & std age of parents:\t", data['age'].mean(), data['age'].std())

# Are parents at more prestigious universities?
print('Non-parents prestige:\t',
      np.mean(subset[subset['first_child_birth'].isna()].prestige_frame))
print('Parents prestige:\t',
      np.mean(subset[~subset['first_child_birth'].isna()].prestige_frame))

print("KS-test:\t", ks_2samp(subset[subset['children'] == 2].prestige_frame,
                             subset[subset['children'] == 1].prestige_frame))
print("Mann-Whitney:\t",
      mannwhitneyu(subset[subset['children'] == 2].prestige_frame,
                   subset[subset['children'] == 1].prestige_frame))


"""
Institutions with better childcare and maternity leave policies have twice(!)
the number of female faculty in STEM
"""
print("How many women & men are at institutions that offered leave?")
offers_parental_leave = sum(
    (subset[(subset['gender'] == 1) &
     (subset['children_no'] > 0)]).parleave_objective_length_women >= 4)

# More than half of female and male respondents were at university which
# offered at least four weeks paid parental leave to faculty
print(
    offers_parental_leave/len(subset[(subset['gender'] == 1) &
                                     (subset['children_no'] > 0)]),
    (offers_parental_leave, len(subset[(subset['gender'] == 1) &
                                       (subset['children_no'] > 0)])),
    (len(subset[subset.parleave_objective_length_women >= 4].university_name_standard.unique()),
     len(subset.university_name_standard.unique())))

offers_parental_leave = sum(
    (subset[(subset['gender'] == 2) &
     (subset['children_no'] > 0)]).parleave_objective_length_men >= 4)
print(
    offers_parental_leave/len(subset[(subset['gender'] == 2) &
                                     (subset['children_no'] > 0)]),
    (offers_parental_leave, len(subset[(subset['gender'] == 2) &
                                       (subset['children_no'] > 0)])),
    (len(subset[subset.parleave_objective_length_men >= 4].university_name_standard.unique()),
     len(subset.university_name_standard.unique())))

print("Z-test:\t", proportions_ztest(
  [sum((subset[(subset['gender'] == 1) &
       (subset['children_no'] > 0)]).parleave_objective_length_women >= 4),
   sum((subset[(subset['gender'] == 2) &
       (subset['children_no'] > 0)]).parleave_objective_length_men >= 4)],
  [len(subset[(subset['gender'] == 1) & (subset['children_no'] > 0)]),
   len(subset[(subset['gender'] == 2) & (subset['children_no'] > 0)])]))


print("No differences in the ages at which men and women become parents.")
print("Women:\t", data[data.gender == 1]['age'].mean(),
      "\tMen:\t", data[data.gender == 2]['age'].mean())
print("T-test:\t", ttest_ind(data[data.gender == 1]['age'].dropna(),
                             data[data.gender == 2]['age'].dropna()))

# Predictive model for parents' age at first kid
rows = []
for _, person in subset.iterrows():
    if np.isnan(person['first_child_birth']) or \
       np.isnan(person['age (actual)']):
        continue

    sid = person['sid']
    p_birth_year = 2017.0 - person['age (actual)']
    k_birth_year = person['first_child_birth']
    age_at_birth = k_birth_year - p_birth_year
    if age_at_birth < 0:
        # One respondent who said the had a 1 yo versus born in 2016
        continue

    tt_start_year = person['first_asst_job_year']
    if ((person['first_asst_job_year'] - p_birth_year) < 0) or \
       np.isnan(tt_start_year):
        continue  # Why?
    if int(person['gender']) == 1:
        gender = 'F'
    elif int(person['gender']) == 2:
        gender = 'M'
    else:
        continue  # Primarily considering men and women

    prestige_current = person['prestige_frame']
    if np.isnan(prestige_current):
        continue

    # Race
    white = person['white']
    hispanic = person['hisp']
    black = person['black']
    asian = person['asian']
    native = person['native']
    hawaiian = person['hawaii']
    other = person['otherace']

    # Other things we might think correlate:
    work_hours = person['workhours']
    parental_stigma = np.mean([person['parstigma1'], person['parstigma2']])
    service = person['service_nc']

    # Parents' education and employment status
    mothers_employment = None
    mothers_education = None
    fathers_employment = None
    fathers_education = None
    if person['p1_gender'] == 1:
        mothers_employment = person['p1_empl']
        mothers_education = person['p1_edu']
        if person['p2_gender'] == 2:  # (Mother, Father)
            fathers_employment = person['p2_empl']
            fathers_education = person['p2_edu']
        elif person['p2_gender'] == 1:  # (Mother, Mother)
            mothers_employment = np.max([person['p1_empl'], person['p2_empl']])
            mothers_education = np.max([person['p1_edu'], person['p2_edu']])
    elif person['p1_gender'] == 2:
        fathers_employment = person['p2_empl']
        fathers_education = person['p2_edu']
        if person['p2_gender'] == 1:  # (Father, Mother)
            mothers_employment = person['p1_empl']
            mothers_education = person['p1_edu']
        elif person['p2_gender'] == 2:  # (Father, Father)
            fathers_employment = np.max([person['p1_empl'], person['p2_empl']])
            fathers_education = np.max([person['p1_edu'], person['p2_edu']])

    # Replace missing values
    if mothers_education in [9, 10]:
        mothers_education = None
    if fathers_education in [9, 10]:
        fathers_education = None
    if mothers_employment in [5, 6, 0]:
        mothers_employment = None
    if fathers_employment in [5, 6, 0]:
        fathers_employment = None

    # Department parental leave
    men_parental_leave = person['parleave_objective_length_men']
    women_parental_leave = person['parleave_objective_length_women']

    # Age
    age_actual = person['age (actual)']
    if np.isnan(age_actual):
        continue

    if (type(person['dblp_pubs']) is float) or (person['dblp_pubs'] is None):
        continue
    pubs = list(person['dblp_pubs'])

    first_child_birth = person['first_child_birth']
    if (first_child_birth == '{}') or np.isnan(first_child_birth) or \
       (first_child_birth >= 2017):
        continue

    rows.append(
        [person['name'], age_at_birth, k_birth_year, p_birth_year,
         tt_start_year, gender, mothers_education, mothers_employment,
         fathers_education, fathers_employment, prestige_current, work_hours,
         parental_stigma, white, hispanic, black, asian, native, hawaiian,
         other, service, first_child_birth, men_parental_leave,
         women_parental_leave, sid, age_actual, pubs])

df_treated = pd.DataFrame(
    rows, columns=['name', 'age_at_birth', 'k_birth_year', 'p_birth_year',
                   'first_asst_job_year', 'gender', 'mothers_education',
                   'mothers_employment', 'fathers_education',
                   'fathers_employment', 'prestige_frame', 'work_hours',
                   'parental_stigma', 'white', 'hispanic', 'black', 'asian',
                   'native', 'hawaiian', 'other', 'service',
                   'first_child_birth', 'parleave_objective_length_men',
                   'parleave_objective_length_women', 'sid', 'age (actual)',
                   'dblp_pubs'])
print(len(df_treated))

print("Relationship between child birth year and parent birth year.")
formula = 'k_birth_year ~ p_birth_year'
mod = smf.ols(formula=formula, data=df_treated)
res = mod.fit()
print(res.summary())

df_treated.loc[:, 'birth_relative_to_career'] = df_treated['k_birth_year'] - df_treated['first_asst_job_year']

if ALLOCATE == "False":
    print("Allocation is false. Stopping here.")
    exit()

# ## Constructing control group
control = subset[subset['first_child_birth'].isna() &
                 (subset['children'] == 1)]
rows = []
for _, person in control.iterrows():
    if np.isnan(person['age (actual)']):
        continue

    sid = person['sid']
    age_actual = person['age (actual)']
    if np.isnan(age_actual):
        continue

    p_birth_year = 2017.0 - age_actual
    tt_start_year = person['first_asst_job_year']
    if tt_start_year == '{}' or np.isnan(tt_start_year) or \
       (person['first_asst_job_year'] - p_birth_year) < 0:
        continue  # Why?

    if person['gender'] == 1:
        gender = 'F'
    elif person['gender'] == 2:
        gender = 'M'
    else:
        continue  # Primarily considering men and women
    prestige_current = person['prestige_frame']
    if np.isnan(prestige_current):
        continue
    prestige_rank = person['prestige_rank']
    if np.isnan(prestige_current):
        continue

    # Race
    white = person['white']
    hispanic = person['hisp']
    black = person['black']
    asian = person['asian']
    native = person['native']
    hawaiian = person['hawaii']
    other = person['otherace']

    # Other things we might think correlate:
    work_hours = person['workhours']
    parental_stigma = np.mean([person['parstigma1'], person['parstigma2']])
    service = person['service_nc']

    # Department parental leave
    men_parental_leave = person['parleave_objective_length_men']
    women_parental_leave = person['parleave_objective_length_women']

    # Age
    age_actual = person['age (actual)']
    first_child_birth = person['first_child_birth']

    # Pubs
    if type(person['dblp_pubs']) is float or (person['dblp_pubs'] is None):
        continue
    pubs = list(person['dblp_pubs'])

    rows.append([person['name'], age_actual, p_birth_year, tt_start_year,
                 gender, prestige_current, work_hours, parental_stigma, white,
                 hispanic, black, asian, native, hawaiian, other, service,
                 pubs, men_parental_leave, women_parental_leave, prestige_rank,
                 first_child_birth, sid])

df_control = pd.DataFrame(
    rows, columns=['name', 'age (actual)', 'p_birth_year',
                   'first_asst_job_year', 'gender', 'prestige_frame',
                   'work_hours', 'parental_stigma', 'white', 'hispanic',
                   'black', 'asian', 'native', 'hawaiian', 'other', 'service',
                   'dblp_pubs', 'parleave_objective_length_men',
                   'parleave_objective_length_women', 'prestige_rank',
                   'first_child_birth', 'sid'])

df_control['univ_offers_leave'] = (df_control.gender == 'M')*(df_control.parleave_objective_length_men) + \
  (df_control.gender == 'F')*(df_control.parleave_objective_length_women)
df_control['univ_offers_leave'] = df_control['univ_offers_leave'] > 4


# Simple linear regression based on parent birth year
# predictions_control = res.predict(df_control)
# predictions_control.hist(color=plot_utils.ALMOST_BLACK)
# df_treated['k_birth_year'].sample(n=100, replace=True).hist(
#     color=plot_utils.ALMOST_BLACK)

# Map prestige values to deciles
pi_values = []
with open('survey_data/faculty_2011/%s_vertexlist.txt' % FIELD) as file:
    pi_reader = csv.DictReader(file, dialect='excel-tab')
    for row in pi_reader:
        pi_values.append(float(row['pi']))

pi_mapping = {}
pi_rankings = pd.cut(pi_values, 10, labels=range(1, 11))
for val, rank in zip(pi_values, pi_rankings):
    pi_mapping[val] = rank

print("Construct control group of women")
control = df_control[(df_control['gender'] == 'F')]
treated = df_treated[(df_treated['gender'] == 'F')]

predictions_control_samples = cohort_utils.allocate_placebo(
    treated, control, iterations=iterations)

df_w_raw_pubs_treated = cohort_utils.compute_publication_trend(
    treated, -5, 10, adjusted=ADJUSTED)

# Drop sensitive variables
df_w_raw_pubs_treated['pre_2000'] = (df_w_raw_pubs_treated['s'] <= MIDPOINT)
df_w_raw_pubs_treated_sub = df_w_raw_pubs_treated[['y', 't', 'c', 'i',
                                                   'i_t', 'round', 'pi',
                                                   'pre_2000']]
df_w_raw_pubs_treated_sub['pi'] = df_w_raw_pubs_treated_sub['pi'].map(pi_mapping)
df_w_raw_pubs_treated_sub['y'] = df_w_raw_pubs_treated_sub['y'].round(4)
df_w_raw_pubs_treated_sub = df_w_raw_pubs_treated.drop(['c'], axis=1)

df_w_raw_pubs_treated_sub.to_csv(
    '../data/treated/%s_publication_outcomes_women_%s_%s.tsv' %
    (FILE_ENDING, FIELD.lower(), DATE),
    sep='\t', na_rep='', index=False)

df_w_raw_pubs_control = cohort_utils.compute_publication_trend(
    control, -5, 10, adjusted=ADJUSTED, control=True,
    predicted_k_birth=predictions_control_samples, iterations=iterations)

# Drop sensitive variables
df_w_raw_pubs_control['pre_2000'] = (df_w_raw_pubs_control['s'] <= MIDPOINT)
df_w_raw_pubs_control_sub = df_w_raw_pubs_control[['y', 't', 'c', 'i',
                                                   'i_t', 'round', 'pi',
                                                   'pre_2000']]
df_w_raw_pubs_control_sub['pi'] = df_w_raw_pubs_control_sub['pi'].map(pi_mapping)
df_w_raw_pubs_control_sub['y'] = df_w_raw_pubs_control_sub['y'].round(4)
df_w_raw_pubs_control_sub = df_w_raw_pubs_control_sub.drop(['c'], axis=1)

df_w_raw_pubs_control_sub.to_csv(
    '../data/control/%s_publication_outcomes_women_%s_%s.tsv' %
    (FILE_ENDING, FIELD.lower(), DATE),
    sep='\t', na_rep='', index=False)

print("Done.\t",
      len(df_w_raw_pubs_treated['i'].unique()),
      len(df_w_raw_pubs_control['i'].unique()))

# Indistinguishable with respect to age at career start
print(
    ttest_ind(
        df_w_raw_pubs_treated[df_w_raw_pubs_treated.c == 0]['age'].dropna(),
        df_w_raw_pubs_control[df_w_raw_pubs_control.c == 0]['age'].dropna(),
        equal_var=False),
    df_w_raw_pubs_treated[df_w_raw_pubs_treated.c == 0]['age'].mean(),
    df_w_raw_pubs_control[df_w_raw_pubs_control.c == 0]['age'].mean())

# Indistinguishable with respect to career year at birth
print(
    ttest_ind(
        df_w_raw_pubs_treated[df_w_raw_pubs_treated.c == 0]['s'].dropna(),
        df_w_raw_pubs_control[df_w_raw_pubs_control.c == 0]['s'].dropna(),
        equal_var=False),
    df_w_raw_pubs_treated[df_w_raw_pubs_treated.c == 0]['s'].mean(),
    df_w_raw_pubs_control[df_w_raw_pubs_control.c == 0]['s'].mean())

# Indistinguishable with respect to career age at birth
print(
    ttest_ind(
        df_w_raw_pubs_treated[df_w_raw_pubs_treated.t == 0]['c'].dropna(),
        df_w_raw_pubs_control[df_w_raw_pubs_control.t == 0]['c'].dropna(),
        equal_var=False),
    df_w_raw_pubs_treated[df_w_raw_pubs_treated.t == 0]['c'].mean(),
    df_w_raw_pubs_control[df_w_raw_pubs_control.t == 0]['c'].mean())

# Indistinguishable with respect to age at first birth
print(
    ttest_ind(
        df_w_raw_pubs_treated[df_w_raw_pubs_treated.t == 0]['age'].dropna(),
        df_w_raw_pubs_control[df_w_raw_pubs_control.t == 0]['age'].dropna(),
        equal_var=False),
    df_w_raw_pubs_treated[df_w_raw_pubs_treated.t == 0]['age'].mean(),
    df_w_raw_pubs_control[df_w_raw_pubs_control.t == 0]['age'].mean())

# Indistinguishable with respect to prestige
print(
    mannwhitneyu(
        df_w_raw_pubs_treated[df_w_raw_pubs_treated.t == 0]['pi'].dropna(),
        df_w_raw_pubs_control[df_w_raw_pubs_control.t == 0]['pi'].dropna()),
    df_w_raw_pubs_treated[df_w_raw_pubs_treated.t == 0]['pi'].mean(),
    df_w_raw_pubs_control[df_w_raw_pubs_control.t == 0]['pi'].mean())

# With respect to productivity?
print(
    ttest_ind(
        df_w_raw_pubs_control[df_w_raw_pubs_control.t == 0]['y'].dropna(),
        df_w_raw_pubs_treated[df_w_raw_pubs_treated.t == 0]['y'].dropna(),
        equal_var=False),
    df_w_raw_pubs_treated[df_w_raw_pubs_treated.t == 0]['y'].mean(),
    df_w_raw_pubs_control[df_w_raw_pubs_control.t == 0]['y'].mean())


# Distribution of biological ages at child birth
# df_w_raw_pubs_control[df_w_raw_pubs_control.t == 0]['age'].hist(density=True)
# df_w_raw_pubs_treated[df_w_raw_pubs_treated.t == 0]['age'].hist(density=True)

# Distribution of career ages at child birth
# df_w_raw_pubs_control[df_w_raw_pubs_control.t == 0]['c'].hist(density=True)
# df_w_raw_pubs_treated[df_w_raw_pubs_treated.t == 0]['c'].hist(density=True)

print("Construct control group of men")
control = df_control[(df_control['gender'] == 'M')]
treated = df_treated[(df_treated['gender'] == 'M')]

predictions_control_samples = cohort_utils.allocate_placebo(
    treated, control, iterations=iterations)

df_m_raw_pubs_treated = cohort_utils.compute_publication_trend(
    treated, -5, 10, adjusted=ADJUSTED)

# Drop sensitive variables
df_m_raw_pubs_treated['pre_2000'] = (df_m_raw_pubs_treated['s'] <= MIDPOINT)
df_m_raw_pubs_treated_sub = df_m_raw_pubs_treated[['y', 't', 'c', 'age', 'i',
                                                   'i_t', 'round', 'pi',
                                                   'pre_2000']]
df_m_raw_pubs_treated_sub['pi'] = df_m_raw_pubs_treated_sub['pi'].map(pi_mapping)
df_m_raw_pubs_treated_sub['y'] = df_m_raw_pubs_treated_sub['y'].round(4)
df_m_raw_pubs_treated_sub = df_m_raw_pubs_treated_sub.drop(['c'], axis=1)

df_m_raw_pubs_treated_sub.to_csv(
    '../data/treated/%s_publication_outcomes_men_%s_%s.tsv' %
    (FILE_ENDING, FIELD.lower(), DATE),
    sep='\t', na_rep='', index=False)

df_m_raw_pubs_control = cohort_utils.compute_publication_trend(
    control, -5, 10, adjusted=ADJUSTED, control=True,
    predicted_k_birth=predictions_control_samples, iterations=iterations)

# Drop sensitive variables
df_m_raw_pubs_control['pre_2000'] = (df_m_raw_pubs_control['s'] <= MIDPOINT)
df_m_raw_pubs_control_sub = df_m_raw_pubs_control[['y', 't', 'c', 'age', 'i',
                                                   'i_t', 'round', 'pi',
                                                   'pre_2000']]
df_m_raw_pubs_control_sub['pi'] = df_m_raw_pubs_control_sub['pi'].map(pi_mapping)
df_m_raw_pubs_control_sub['y'] = df_m_raw_pubs_control_sub['y'].round(4)
df_m_raw_pubs_control_sub = df_m_raw_pubs_control_sub.drop(['c'], axis=1)

df_m_raw_pubs_control_sub.to_csv(
    '../data/control/%s_publication_outcomes_men_%s_%s.tsv' %
    (FILE_ENDING, FIELD.lower(), DATE), sep='\t', na_rep='', index=False)

print("Done.\t", len(df_m_raw_pubs_treated['i'].unique()),
      len(df_m_raw_pubs_control.dropna()['i'].unique()))

# Indistinguishable with respect to age at career start
print(
    ttest_ind(
        df_m_raw_pubs_treated[df_m_raw_pubs_treated.c == 0]['age'].dropna(),
        df_m_raw_pubs_control[df_m_raw_pubs_control.c == 0]['age'].dropna(),
        equal_var=False),
    df_m_raw_pubs_treated[df_m_raw_pubs_treated.c == 0]['age'].mean(),
    df_m_raw_pubs_control[df_m_raw_pubs_control.c == 0]['age'].mean())

# Indistinguishable with respect to career year at birth
print(
    ttest_ind(
        df_m_raw_pubs_treated[df_m_raw_pubs_treated.c == 0]['s'].dropna(),
        df_m_raw_pubs_control[df_m_raw_pubs_control.c == 0]['s'].dropna(),
        equal_var=False),
    df_m_raw_pubs_treated[df_m_raw_pubs_treated.c == 0]['s'].mean(),
    df_m_raw_pubs_control[df_m_raw_pubs_control.c == 0]['s'].mean())

# Indistinguishable with respect to career age at birth
print(
    ttest_ind(
        df_m_raw_pubs_treated[df_m_raw_pubs_treated.t == 0]['c'].dropna(),
        df_m_raw_pubs_control[df_m_raw_pubs_control.t == 0]['c'].dropna(),
        equal_var=False),
    df_m_raw_pubs_treated[df_m_raw_pubs_treated.t == 0]['c'].mean(),
    df_m_raw_pubs_control[df_m_raw_pubs_control.t == 0]['c'].mean())


# Indistinguishable with respect to age at first birth
print(
    ttest_ind(
        df_m_raw_pubs_treated[df_m_raw_pubs_treated.t == 0]['age'].dropna(),
        df_m_raw_pubs_control[df_m_raw_pubs_control.t == 0]['age'].dropna(),
        equal_var=False),
    df_m_raw_pubs_treated[df_m_raw_pubs_treated.t == 0]['age'].mean(),
    df_m_raw_pubs_control[df_m_raw_pubs_control.t == 0]['age'].mean())

# Indistinguishable with respect to prestige
print(mannwhitneyu(
    df_m_raw_pubs_treated[df_m_raw_pubs_treated.t == 0]['pi'].dropna(),
    df_m_raw_pubs_control[df_m_raw_pubs_control.t == 0]['pi'].dropna()),
    df_m_raw_pubs_treated[df_m_raw_pubs_treated.t == 0]['pi'].mean(),
    df_m_raw_pubs_control[df_m_raw_pubs_control.t == 0]['pi'].mean())

# With respect to productivity?
print(
    ttest_ind(
        df_m_raw_pubs_control[df_m_raw_pubs_control.t == 0]['y'].dropna(),
        df_m_raw_pubs_treated[df_m_raw_pubs_treated.t == 0]['y'].dropna(),
        equal_var=False),
    df_m_raw_pubs_treated[df_m_raw_pubs_treated.t == 0]['y'].mean(),
    df_m_raw_pubs_control[df_m_raw_pubs_control.t == 0]['y'].mean())


# Distribution of biological ages at child birth
# df_m_raw_pubs_control[df_m_raw_pubs_control.t == 0]['age'].hist(density=True)
# df_m_raw_pubs_treated[df_m_raw_pubs_treated.t == 0]['age'].hist(density=True)

# Distribution of career ages at child birth
# df_m_raw_pubs_control[df_m_raw_pubs_control.t == 0]['c'].hist(density=True)
# df_m_raw_pubs_treated[df_m_raw_pubs_treated.t == 0]['c'].hist(density=True)

print("Is there a relationship between age at which you become a parent and \
      career age? Yes.")
parents = pd.concat([df_m_raw_pubs_treated, df_w_raw_pubs_treated])

data = parents[parents.t == 0][['c', 'age']]
data.plot(x='c', y='age', kind='scatter', color=plot_utils.ALMOST_BLACK)
data['b'] = data.c - data.age
mod = smf.ols(formula='age ~ c', data=data)
res = mod.fit()
print(res.summary())

print("Is there a relationship between age at which you become a parent and \
      real time? (That is, is age of parenthood changing over time?)")
data = parents[parents.t == 0][['s', 'age']]
data['b'] = data.s - data.age
mod = smf.ols(formula='age ~ s', data=data)
res = mod.fit()
print(res.summary())

# Demonstrate several strategies for alignment
control_groups = {}
strategies = ['bootstrap', 'linear', 'quadratic', 'lognormal']
for strategy in strategies:
    print(strategy)

    control = df_control[(df_control['gender'] == 'M')]
    treated = df_treated[(df_treated['gender'] == 'M')]

    print(control.shape, treated.shape)
    predictions_control_samples = cohort_utils.allocate_placebo(
        treated, control, iterations=iterations, strategy=strategy)
    df_m_control = cohort_utils.compute_publication_trend(
        control, -5, 10, adjusted=ADJUSTED, control=True,
        predicted_k_birth=predictions_control_samples, iterations=iterations)

    control = df_control[(df_control['gender'] == 'F')]
    treated = df_treated[(df_treated['gender'] == 'F')]

    print(control.shape, treated.shape)
    predictions_control_samples = cohort_utils.allocate_placebo(
        treated, control, iterations=iterations, strategy=strategy)
    df_w_control = cohort_utils.compute_publication_trend(
        control, -5, 10, adjusted=ADJUSTED, control=True,
        predicted_k_birth=predictions_control_samples, iterations=iterations)

    control_groups[strategy] = pd.concat([df_m_control, df_w_control])

start = 1965
end = 2015
parents = pd.concat([df_m, df_w], axis=0, ignore_index=False)

mean_age_birth_true = []
mean_age_birth_predicted = {}
for (lwr, upr) in zip(list(range(start, end, 1)),
                      list(range(start+1, end+1, 1))):
    ids_bin = parents[parents['t'] == 0]
    ids_bin = ids_bin[((ids_bin.s) < upr) & ((ids_bin.s) >= lwr)]['i'].unique()

    bin_parents = parents.loc[parents['i'].isin(ids_bin)]
    mean_age_birth_true.append(bin_parents[bin_parents['t'] == 0]['age'].mean(
        skipna=True))

    for strategy in control_groups.keys():
        ids_bin = control_groups[strategy][control_groups[strategy]['t'] == 0]
        ids_bin = ids_bin[((ids_bin.s) < upr) &
                          ((ids_bin.s) >= lwr)]['i'].unique()

        bin_parents = control_groups[strategy].loc[control_groups[strategy]['i'].isin(ids_bin)]
        if strategy not in mean_age_birth_predicted:
            mean_age_birth_predicted[strategy] = []
        mean_age_birth_predicted[strategy].append(
            bin_parents[bin_parents['t'] == 0]['age'].mean(skipna=True))


fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(12, 3), sharex=True,
                       sharey=True, dpi=500)
strategies = ['linear', 'lognormal', 'bootstrap']
for i in range(3):
    ax[i].scatter(parents[parents.t == 0].s, parents[parents.t == 0]['age'],
                  color=plot_utils.ALMOST_BLACK, alpha=0.1, edgecolors=None,
                  label='All Parents')
    ax[i].plot(list(range(start, end+3, 1)),
               res.predict(pd.DataFrame({'s': list(range(start, end+3, 1))})),
               linestyle='--', linewidth=1, zorder=2, label='Trend',
               color=plot_utils.ALMOST_BLACK)

    ax[i].scatter(list(range(start, end, 1)), mean_age_birth_true,
                  color=plot_utils.ALMOST_BLACK, edgecolors=None, zorder=2,
                  label='Avg. Parents')
    ax[i].scatter(list(range(start, end, 1)),
                  mean_age_birth_predicted[strategies[i]],
                  color=plot_utils.ACCENT_COLOR, zorder=2, label='Prediction')
    ax[i].set_title(strategies[i].title(), fontsize=plot_utils.TITLE_SIZE)

ax[0].set_ylabel('Average Age of Parent', fontsize=plot_utils.LABEL_SIZE)
ax[0].set_xlabel('Child Birth Year', fontsize=plot_utils.LABEL_SIZE)
ax[0].set_xlim(start-2, end+2)

ax[0].legend(frameon=False, fontsize=plot_utils.LEGEND_SIZE-2, ncol=1)

plt.tight_layout()
plot_utils.finalize(ax[0])
plot_utils.finalize(ax[1])
plot_utils.finalize(ax[2])
plt.savefig(
    '../plots/descriptive/parent_age_over_time_distribution_%s_%s.pdf' %
    (FIELD.lower(), DATE), dpi=1000)
