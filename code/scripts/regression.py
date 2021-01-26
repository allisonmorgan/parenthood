import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np
import pandas as pd
import scipy.stats

# Pull out a particular coefficient from the regression
LABEL = 'i_t[T.1]:C(t)[%d]'
DD_LABEL = 'treated*group'

# Range of productivity measure (in terms of kid time)
T = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
S = np.arange(1935, 1995)  # Range of parents' birth years


def generate_outcomes(N, penalty_length=0.0, penalty_size=1.0):
    """
    Develop N data points for Y_{i, s, t}^{g} across all {i, s, t}.
    Penalty parameter pushes Y lower at kid's birth (t = 0)
    """
    rows = []
    for i in range(N):
        # Generate random productivity sequence.
        rate = 2.0
        penalty = penalty_size

        productivity = np.random.poisson(rate, len(T))
        if penalty_length > 0:
            delta_rate = (1.0 - penalty_size)/penalty_length
            for d in range(int(penalty_length)):
                productivity[T.index(0) + d] = np.random.poisson(rate*penalty,
                                                                 1)
                penalty = penalty + delta_rate
        y = productivity  # np.cumsum(productivity)

        # Pick parents' age (uniform) & first kid's birth year (normal)
        p_birth_year = np.random.choice(S)
        k_birth_year = 16 + int(np.random.poisson(5, 1))

        # Age at kid's birth across the event time
        for j, t in enumerate(T):
            s = p_birth_year + k_birth_year + t  # Current year
            a = s - p_birth_year

            # Add dummy variable with respect to t = -1
            i_t = 0 if t == -1 else 1

            # Add the covariates of a particular Y_{i, s, t}^{g}
            rows.append([y[j], t, a, s, i_t, i])

    # Add these N individuals, with their trajectories
    df = pd.DataFrame(rows, columns=['y', 't', 'age', 's', 'i_t', 'i'])
    return df


def build_model(df, t_val=None):
    if t_val is None:
        # Generate regression formula
        data = df
        formula = 'y ~ i_t:C(t) + C(c) + pi + 1'
        data = sm.add_constant(data)
    else:
        # Subset to particular t and build new model
        data = df[df['t'] == t_val]
        formula = 'y ~ C(c) + pi + 1'

    # Fit an OLS model
    mod = smf.ols(formula=formula, data=data)
    res = mod.fit()

    return res


def raw_difference(df, t_1, t_2, cumulative=True):
    # For each person, calculate the difference: Y_{i, t_1} - Y_{i, t_2}
    differences = []
    indices = np.unique(df['i'])

    for i in indices:
        person = df[df['i'] == i]
        cumulative_sum = np.cumsum(person['y'].values)
        try:
            delta = person[person['t'] == t_2]['y'].values[0] \
                     - person[person['t'] == t_1]['y'].values[0]
            if cumulative and len(cumulative_sum) == len(T):
                delta = cumulative_sum[T.index(t_2)] \
                         - cumulative_sum[T.index(t_1)]
            if not np.isnan(delta):
                differences.append(delta)
        except:
            # This may potentially print out quite often
            # print("Missing value for t_1 or t_2")
            continue
    return differences


def difference(df_1, df_2, t_1, t_2, cumulative=True):
    control = df_1.copy()
    control['treated'] = np.repeat(0, len(control))

    treated = df_2.copy()
    treated['treated'] = np.repeat(1, len(treated))

    formula = 'y ~ t + C(treated) + T*C(treated)'
    if cumulative:
        cumulative_sum = []
        for i in control['i'].unique():
            cumulative_sum.extend(np.cumsum(control[control['i'] == i]['y']))
        control['cumulative_y'] = cumulative_sum

        cumulative_sum = []
        for i in treated['i'].unique():
            cumulative_sum.extend(np.cumsum(treated[treated['i'] == i]['y']))
        treated['cumulative_y'] = cumulative_sum

        formula = 'cumulative_y ~ t + C(treated) + T*C(treated)'

    control = control.loc[(control['t'] == t_1) | (control['t'] == t_2)]
    control.dropna(inplace=True)
    control['T'] = np.where(control['t'] == t_1, 0, 1)

    treated = treated.loc[(treated['t'] == t_1) | (treated['t'] == t_2)]
    treated.dropna(inplace=True)
    treated['T'] = np.where(treated['t'] == t_1, 0, 1)

    data = control.append(treated, ignore_index=True, sort=False)
    # data = sm.add_constant(data)

    mod = smf.ols(formula=formula, data=data)
    res = mod.fit()

    # Difference is on the interatcion term: t*C(treated)
    return res


def get_sample(df):
    n = df['i'].unique()
    # Generate a sample with replacement
    sample_ids = np.random.choice(n, len(n), replace=True)
    sample = pd.DataFrame(columns=df.columns)
    sample = sample.append(df[df['i'].isin(sample_ids)],
                           ignore_index=True,
                           sort=False)
    # Return just the one sample
    return sample


def get_bootstrap_interval(df, N):
    rel_productivity = [None]*N
    for i in range(N):
        # Consider one of our samples
        sample = get_sample(df)

        # Build linear model across time for this sample
        mod = build_model(sample, t_val=None)
        alpha = [mod.params[LABEL % t] for t in T]

        # Calculate the expectation of the sample across time
        expectation = [np.mean(build_model(sample, t_val=t).predict())
                       for t in T]
        rel_productivity[i] = [alpha[j]/expectation[j] for j in range(len(T))]

    # return np.std(rel_productivity, axis=0)
    return scipy.stats.sem(rel_productivity, axis=0)


def get_bootstrap_trajectories(df, N, cumulative=True):
    productivity_trajectory = [None]*N
    for i in range(N):
        # Consider one of our samples
        sample = get_sample(df)

        # Get the average trajectory of the sample
        if cumulative:
            productivity_trajectory[i] = np.cumsum(
                [sample[sample['t'] == t]['y'].mean() for t in T])
        else:
            productivity_trajectory[i] = [sample[sample['t'] == t]['y'].mean()
                                          for t in T]

    # return scipy.stats.sem(productivity_trajectory, axis=0)
    return scipy.stats.sem(productivity_trajectory, axis=0)


def construct_empirical_CI(df, alpha=0.05, cumulative=True):
    df_copy = df.copy(deep=True)
    df_copy['cumulative'] = df_copy.groupby(['i', 'round'])['y'].cumsum()

    lower = df_copy.groupby(['t'])['cumulative'].quantile(q=alpha/2.0)
    upper = df_copy.groupby(['t'])['cumulative'].quantile(q=1.0-(alpha/2.0))

    return(lower.to_list(), upper.to_list())
