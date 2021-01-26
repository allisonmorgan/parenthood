#!/usr/bin/env python

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Fuzzy string matching for first/middle/last authorship detection
from fuzzywuzzy import fuzz


# DBLP Adjustments from "The misleading narrative..."
GROW_SLOPE = 0.131873
GROW_INTER = -258.286620
DBLP_SLOPE = 0.010588
DBLP_INTER = -20.434804


def grow_adjust(x):
    # GROW correction
    return (2018*GROW_SLOPE+GROW_INTER)/(x*GROW_SLOPE+GROW_INTER)


def dblp_adjust(x):
    # DBLP correction
    return (2018*DBLP_SLOPE+DBLP_INTER)/(x*DBLP_SLOPE+DBLP_INTER)


def adjust(x):
    return (grow_adjust(x) * dblp_adjust(x))


# Authorship roles
FAP = 0
MAP = 1
LAP = 2
author_roles = ['First', 'Middle', 'Last']


def get_author_role(faculty_name, author_list):
    """
    Detect author position from faculty name and the list of authors.
    Position type returned as a string: "first", "middle", or "last".
    """
    author_sim = [fuzz.ratio(faculty_name.lower(), name.lower())
                  for name in author_list]
    position = np.argmax(author_sim)
    if position == 0:
        return FAP
    elif position == len(author_sim) - 1:
        return LAP
    else:
        return MAP


strategies = ['bootstrap', 'linear', 'quadratic', 'lognormal']


def allocate_placebo(treated, control, iterations=500, strategy='bootstrap'):
    if strategy not in strategies:
        raise NotImplementedError

    if strategy == 'bootstrap':
        return bootstrap_allocation(treated, control, iterations)
    elif strategy == 'linear':
        return linear_allocation(treated, control, iterations)
    elif strategy == 'quadratic':
        return quadratic_allocation(treated, control, iterations)
    elif strategy == 'lognormal':
        return lognormal_allocation(treated, control, iterations)


def bootstrap_allocation(treated, control, iterations):
    predictions = []
    for i, row in control.iterrows():
        parent_birth_year = row.p_birth_year
        delta = 0

        less_than_parent = (treated.p_birth_year <=
                            (parent_birth_year + delta))
        greater_than_parent = (treated.p_birth_year >=
                               (parent_birth_year - delta))

        comparison_group = treated[less_than_parent & greater_than_parent]
        if len(comparison_group) == 0:
            sample_pred = [np.nan]*iterations
        else:
            sample_pred = comparison_group.k_birth_year.sample(
                    n=iterations, replace=True).values

        predictions.append(sample_pred)

    return np.array(predictions).transpose()


def linear_allocation(treated, control, iterations):
    predictions = []
    for i in range(iterations):
        # Build a model for a subset of the treated data
        sampled = treated.sample(n=len(treated), replace=True, random_state=i)
        formula = 'k_birth_year ~ p_birth_year + gender + first_asst_job_year + prestige_frame'
        mod = smf.ols(formula=formula, data=sampled)
        res = mod.fit()

        # Return the predictions of this model
        # for the control group
        predictions.append(
            [np.round(each) for each in list(res.predict(control))])

    return predictions


def quadratic_allocation(treated, control, iterations):
    predictions = []
    for i in range(iterations):
        # Build a model for a subset of the treated data
        sampled = treated.sample(n=len(treated), replace=True, random_state=i)
        formula = 'k_birth_year ~ p_birth_year + p_birth_year**2 + gender + first_asst_job_year + prestige_frame'
        mod = smf.ols(formula=formula, data=sampled)
        res = mod.fit()

        # Return the predictions of this model for the control group
        predictions.append(
            [np.round(each) for each in list(res.predict(control))])

    return predictions


def lognormal_allocation(treated, control, iterations):
    predictions = []
    for _, row in control.iterrows():
        parent_birth_year = row.p_birth_year
        delta = 0

        less_than_parent = (treated.p_birth_year <=
                            (parent_birth_year + delta))
        greater_than_parent = (treated.p_birth_year >=
                               (parent_birth_year - delta))

        comparison_group = treated[less_than_parent & greater_than_parent]
        if len(comparison_group) == 0:
            sample_age_preds = [np.nan]*iterations
        else:
            sample_age_preds = np.random.lognormal(
                mean=(comparison_group.k_birth_year -
                      comparison_group.p_birth_year).mean(),
                sigma=(comparison_group.k_birth_year -
                       comparison_group.p_birth_year).std(),
                size=iterations)

        predictions.append(
            [(parent_birth_year + np.round(np.log(pred)))
             for pred in sample_age_preds])

    return np.array(predictions).transpose()


def construct_covariates(trend, person, k_birth_year, i,
                         time, iteration, id_key):
    p_birth_year = 2017.0 - person['age (actual)']  # Survey was run in 2017

    if person['prestige_frame'] is None:
        return []
    pi = person['prestige_frame']

    tt_start_year = person['first_asst_job_year']
    if (tt_start_year == '{}') or np.isnan(tt_start_year) or \
       (person['first_asst_job_year'] - p_birth_year) < 0:
        return []

    sid = person[id_key]

    # Add that productivity, career age, and event time to a data frame
    rows = []
    for j, t in enumerate(time):
        if j >= len(trend):
            break
        s = k_birth_year + t  # Current year
        a = (k_birth_year + t) - p_birth_year  # Current age
        c = (k_birth_year + t) - tt_start_year  # Career age
        if a < 0:
            continue  # Ages should be positive
            # TODO: One person who listed row['first_child_birth'] as 1.0.
            # Likely they had a one year old during survey, and didn't
            # list year of birth like others.

        # Add dummy variable with respect to t = -1
        i_t = 0 if t == -1 else 1

        # Add the covariates of a particular Y_{i, s, t}^{g}
        rows.append([trend[j], t, a, s, c, i_t, i, pi, iteration, sid])

    return rows


def compute_publication_trend(cohort, t_lower, t_upper, adjusted=True,
                              author_position=None, control=False,
                              predicted_k_birth=None, iterations=1,
                              relative_to='first_child_birth', id_key='sid'):
    rows = []
    # Iterate through the population. For those with kids, count how many
    # papers each person has in a given year from the range [t_lower, t_upper].
    for count, (i, row) in enumerate(cohort.iterrows()):
        if np.isnan(row['age (actual)']) or (row['dblp_pubs'] is np.nan) or \
         (row['dblp_pubs'] is None):
            continue

        k_birth_years = []
        if not control:
            k_birth_years = [row[relative_to]]
        else:
            for iteration in range(iterations):
                k_birth_years.append(predicted_k_birth[iteration][count])

        # If this year is in the future (relative to the survey date), then we
        # cannot consider this person in the control group.
        if np.any([(k_birth_year == '{}') or np.isnan(k_birth_year) or
                   (k_birth_year >= 2017) for k_birth_year in k_birth_years]):
            continue

        # Name used for finding first/last author publication trends
        author_name = row['name']

        # Get productivity
        for iteration, k_birth_year in enumerate(k_birth_years):
            trend = []
            T = range(t_lower, t_upper + 1)

            for j in T:
                yearly_pubs = 0
                if author_position is None:
                    pubs = [entry[0] for entry in row['dblp_pubs']]
                    yearly_pubs = np.count_nonzero(
                        np.array(pubs) == (k_birth_year + j))
                elif author_position == "last":
                    la_pubs = [entry[0] for entry in row['dblp_pubs'] if
                               get_author_role(author_name, entry[1]) == LAP]
                    yearly_pubs = np.count_nonzero(
                        np.array(la_pubs) == (k_birth_year + j))
                elif author_position == "first":
                    fa_pubs = [entry[0] for entry in row['dblp_pubs'] if
                               get_author_role(author_name, entry[1]) == FAP]
                    yearly_pubs = np.count_nonzero(
                        np.array(fa_pubs) == (k_birth_year + j))

                if (k_birth_year + j) >= 2020:
                    trend.append(np.nan)
                    continue

                adjustment = 1
                if adjusted:
                    adjustment = adjust(k_birth_year + j)

                trend.append(yearly_pubs*adjustment)

            rows.extend(construct_covariates(trend, row, k_birth_year,
                                             i, T, iteration, id_key))

    return pd.DataFrame(rows, columns=['y', 't', 'age', 's', 'c', 'i_t',
                                       'i', 'pi', 'round', 'sid'])


def compute_coauthor_trend(cohort, t_lower, t_upper, control=False,
                           predicted_k_birth=None, iterations=1,
                           relative_to='first_child_birth'):
    rows = []
    # Iterate through the population. For those with kids, count how many
    # papers each person has in a given year from the range [t_lower, t_upper]
    for count, (i, row) in enumerate(cohort.iterrows()):
        if np.isnan(row['age (actual)']) or (row['dblp_pubs'] is np.nan) or \
         (row['dblp_pubs'] is None):
            continue

        k_birth_years = []
        if not control:
            k_birth_years = [row[relative_to]]
        else:
            for iteration in range(iterations):
                k_birth_years.append(predicted_k_birth[iteration][count])

        # If this year is in the future (relative to the survey date), then we
        # cannot consider this person in the control group.
        if np.any([(k_birth_year >= 2017) or np.isnan(k_birth_year)
                   for k_birth_year in k_birth_years]):
            continue

        # Get coauthors
        post_kid_coauths = {}
        for entry in row['dblp_pubs']:
            if not entry[0] in post_kid_coauths:
                post_kid_coauths[entry[0]] = []
            post_kid_coauths[entry[0]].extend(entry[1])

        # Number of new unique coauthors over time
        total_coauth_set = set()
        uniq_coauth_trend = []

        for k, (year, coauths) in enumerate(sorted(post_kid_coauths.items())):
            n_new_coauths = len(set(coauths).difference(total_coauth_set))
            if k == 0:
                # Subtract off the person from their coauthor list
                n_new_coauths = n_new_coauths - 1
            uniq_coauth_trend.append((year, n_new_coauths))
            total_coauth_set.update(set(coauths))

        # Trend relative to child's birth year
        for iteration, k_birth_year in enumerate(k_birth_years):
            trend = []
            T = range(t_lower, t_upper + 1)

            for j in T:
                yearly_coauths = 0
                if (k_birth_year + j) in dict(uniq_coauth_trend):
                    yearly_coauths = dict(uniq_coauth_trend)[k_birth_year + j]

                if (k_birth_year + j) >= 2019:
                    yearly_coauths = np.nan

                trend.append(yearly_coauths)

            rows.extend(construct_covariates(trend, row, k_birth_year,
                                             i, T, iteration))

    return pd.DataFrame(rows, columns=['y', 't', 'age', 's', 'c', 'i_t',
                                       'i', 'pi', 'round'])


def generate_average_cumulative(df, agg='mean'):
    """
    Compute the average / median, cumulative outcome (y)
    """
    if agg not in ['mean', 'median']:
        raise NotImplementedError

    df_copy = df.copy(deep=True)
    df_copy['cumulative'] = df_copy.groupby(['i', 'round'])['y'].cumsum()
    if agg == 'median':
        return df_copy.groupby(['t'])['cumulative'].median()
    return df_copy.groupby(['t'])['cumulative'].mean()
