
import pandas as pd
import numpy as np
import csv

from scripts.parse.institution_parser import INST_NAME_ALIASES

BUSI_HIS_RESPONSES = '../data/survey_data/his_busi_survey/main_sep2019.xlsx'
HIS_FRAME_HEADS = '../data/survey_data/his_busi_survey/HIS_intro_2018_10_22b_unlocked.xlsx'
HIS_FRAME_ALL = '../data/survey_data/his_busi_survey/HIS_allinvited_exceptheads.xls'
BUSI_FRAME_HEADS = '../data/survey_data/his_busi_survey/intro_sep2019_unlocked.xlsx'
BUSI_FRAME_ALL = '../data/survey_data/his_busi_survey/BUSI_participant_codes_send1.xls'
BUSI_HIS_PUBS = '../data/survey_data/busi_his_pubs/hand_coded_pubs_complete_list_2-21-2020.json'

CS_RESPONSES = '../data/survey_data/cs_survey/data_15oct2017_final2f.xlsx'
CS_FRAME = '../data/survey_data/cs_survey/frame_jun8_2018.xlsx'
CS_PUBS = '../data/survey_data/cs_pubs/cs_prior_productivity_authorship_feb5_2020.json'

PRESTIGE = '../data/survey_data/faculty_2011/%s_vertexlist.txt'
PARENTAL_LEAVE = '../data/survey_data/parental_leave/parental_leave_policies_apr_2018.tsv'

codebook_age = dict(list(zip(range(1, 82, 1), range(1996, 1915, -1))))
codebook_age[-77] = None
codebook_age[0] = None
codebook_age[np.nan] = None


def load_all_faculty():
    # Read in Business / History responses
    busi_his_merged = load_business_history_faculty()

    # Read in the CS publications and responses
    cs_merged = load_cs_faculty()

    # Merge all the responses together!
    df = pd.concat([busi_his_merged, cs_merged], axis=0, sort=False)

    # This child age field is a bit messy. Since we focus on first child,
    # these aren't too big a deal.
    df.loc[df.chage1 == 200, 'chage1'] = 2000  # We should fix these.
    df.loc[df.chage1 == 199, 'chage1'] = 1999
    df.loc[df.chage1 == 1900, 'chage1'] = np.nan
    df.loc[df.chage1 == 69, 'chage1'] = 1969
    df.loc[df.chage1 == 93, 'chage1'] = 1993
    df.loc[df.chage1 == 61, 'chage1'] = 1961

    df.loc[df.chage2 == 218, 'chage2'] = 2018  # We should fix these.
    df.loc[df.chage2 == 199, 'chage2'] = 1990
    df.loc[df.chage2 == 69, 'chage2'] = 1969
    df.loc[df.chage2 == 95, 'chage2'] = 1995
    df.loc[df.chage2 == 19885, 'chage2'] = 1985
    df.loc[df.chage2 == 98, 'chage2'] = 1998
    df.loc[df.chage2 == 78, 'chage2'] = 1978
    df.loc[df.chage2 == 2973, 'chage2'] = 1973
    df.loc[df.chage2 == 70, 'chage2'] = 1970

    df.loc[df.chage3 == 74, 'chage3'] = 1974  # We should fix these.
    df.loc[df.chage3 == 20016, 'chage3'] = 2016
    df.loc[df.chage3 == 72, 'chage3'] = 1972
    df.loc[df.chage3 == 81, 'chage3'] = 1981
    df.loc[df.chage3 == 97, 'chage3'] = 1997
    df.loc[df.chage3 == 2, 'chage3'] = np.nan
    df.loc[df.chage5 == 1, 'chage5'] = np.nan

    # Calculate the year of each parent's first child birth (e.g., the year they became parents).
    first_child_age = []
    for i, row in df.iterrows():
        first_child_age.append(np.min(row[['chage1', 'chage2', 'chage3',
                                           'chage4', 'chage5', 'chage6',
                                           'chage7', 'chage8', 'chage9']]))
    df['first_child_birth'] = first_child_age

    # How old is their youngest child at the time of the survey?
    df['youngest'] = 2017 - df.loc[:, ['chage1', 'chage2', 'chage3', 'chage3',
                                       'chage4', 'chage5', 'chage6', 'chage7',
                                       'chage8', 'chage9']].max(axis=1)

    df['haskiddo'] = df.youngest < 5
    df['hasunder10'] = df.youngest < 10

    # How many papers do faculty aim to publish
    df['aim_max'] = pd.to_numeric(df.aim_max, errors='coerce')
    df['aim_min'] = pd.to_numeric(df.aim_min, errors='coerce')

    df.loc[df.aim_max < 0, 'aim_max'] = np.nan
    df.loc[df.aim_min < 0, 'aim_min'] = np.nan
    df['aim_avg'] = .5*(df.aim_max + df.aim_min)

    # Expectations and likely productivity levels of different groups
    df['desnorm_wchild'] = pd.to_numeric(df['desnorm_wchild'], errors='coerce')
    df['desnorm_wnochild'] = pd.to_numeric(df['desnorm_wnochild'],
                                           errors='coerce')
    df['desnorm_mchild'] = pd.to_numeric(df['desnorm_mchild'], errors='coerce')
    df['desnorm_mnochild'] = pd.to_numeric(df['desnorm_mnochild'],
                                           errors='coerce')

    df['injnorm_wchild'] = pd.to_numeric(df['injnorm_wchild'], errors='coerce')
    df['injnorm_wnochild'] = pd.to_numeric(df['injnorm_wnochild'],
                                           errors='coerce')
    df['injnorm_mchild'] = pd.to_numeric(df['injnorm_mchild'], errors='coerce')
    df['injnorm_mnochild'] = pd.to_numeric(df['injnorm_mnochild'],
                                           errors='coerce')

    df.loc[df.desnorm_wchild < 0, 'desnorm_wchild'] = np.nan
    df.loc[df.desnorm_wnochild < 0, 'desnorm_wnochild'] = np.nan
    df.loc[df.desnorm_mchild < 0, 'desnorm_mchild'] = np.nan
    df.loc[df.desnorm_mnochild < 0, 'desnorm_mnochild'] = np.nan

    df.loc[df.injnorm_wchild < 0, 'injnorm_wchild'] = np.nan
    df.loc[df.injnorm_wnochild < 0, 'injnorm_wnochild'] = np.nan
    df.loc[df.injnorm_mchild < 0, 'injnorm_mchild'] = np.nan
    df.loc[df.injnorm_mnochild < 0, 'injnorm_mnochild'] = np.nan

    return df


def load_business_history_faculty():
    # Read in history data
    responses = pd.read_excel(BUSI_HIS_RESPONSES)
    responses.shape

    responses = responses[responses.lastpage != 0]
    responses_consent = responses[responses.consent == 1]
    responses_consent.shape

    his_ids = pd.read_excel(HIS_FRAME_HEADS)
    his_ids_more = pd.read_excel(HIS_FRAME_ALL)

    his_merged_on_dept_heads = pd.merge(his_ids, responses_consent,
                                        left_on='session_id',
                                        right_on='pid', how='inner')

    his_merged_on_dept_heads['u_name'] = his_merged_on_dept_heads['firstname'] + ' ' + his_merged_on_dept_heads['lastname']
    his_merged_on_dept_heads['u_university'] = his_merged_on_dept_heads['university']
    his_merged_on_dept_heads['u_email'] = his_merged_on_dept_heads['email']
    his_merged_on_dept_heads['u_department'] = his_merged_on_dept_heads['department']

    his_merged_on_invited_his = pd.merge(his_ids_more, responses_consent,
                                         left_on='code', right_on='pid',
                                         how='inner')

    all_history = pd.concat([his_merged_on_dept_heads,
                             his_merged_on_invited_his], axis=0, sort=False)

    # Read in business data
    busi_ids = pd.read_excel(BUSI_FRAME_HEADS)
    busi_ids_more = pd.read_excel(BUSI_FRAME_ALL)

    busi_merged_on_dept_heads = pd.merge(busi_ids, responses_consent,
                                         left_on='session_id', right_on='pid',
                                         how='inner')

    busi_merged_on_dept_heads['u_name'] = busi_merged_on_dept_heads['firstname'] + ' ' + busi_merged_on_dept_heads['lastname']
    busi_merged_on_dept_heads['u_university'] = busi_merged_on_dept_heads['university']
    busi_merged_on_dept_heads['u_email'] = busi_merged_on_dept_heads['email']
    busi_merged_on_dept_heads['u_department'] = busi_merged_on_dept_heads['department']

    busi_merged_on_invited = pd.merge(busi_ids_more, responses_consent,
                                      left_on='code', right_on='pid',
                                      how='inner')

    all_business = pd.concat(
        [busi_merged_on_dept_heads, busi_merged_on_invited], axis=0,
        sort=False)
    # History responses were in the business responses file
    all_business = all_business[all_business.pid.isin(all_history.pid) == False]

    # Standardize university names
    all_business['university_name_standard'] = all_business['u_university'].apply(
        lambda x: INST_NAME_ALIASES[x] if x in INST_NAME_ALIASES else x)
    all_history['university_name_standard'] = all_history['u_university'].apply(
        lambda x: INST_NAME_ALIASES[x] if x in INST_NAME_ALIASES else x)

    for field, field_data in [('Business', all_business), ('History', all_history)]:
        pi_rank_mapping = {}
        parental_leave_mapping = {}

        # Extract prestige / ranking data and parental leave information
        with open(PRESTIGE % field) as rankings, \
             open(PARENTAL_LEAVE) as parental_leave:

            pi_reader = csv.DictReader(rankings, dialect='excel-tab')
            for row in pi_reader:
                pi_rank_mapping[row['institution']] = row

            leave_reader = csv.DictReader(parental_leave, dialect='excel-tab')
            for row in leave_reader:
                parental_leave_mapping[row['university_name']] = row

        field_data['prestige_inv'] = field_data['university_name_standard'].apply(
            lambda x: pi_rank_mapping[x]['pi'] if x in pi_rank_mapping else np.nan)
        field_data['prestige_rank_inv'] = field_data['university_name_standard'].apply(
            lambda x: pi_rank_mapping[x]['# u'] if x in pi_rank_mapping else np.nan)

        field_data['prestige_inv'] = pd.to_numeric(field_data['prestige_inv'])
        field_data['prestige_rank_inv'] = pd.to_numeric(field_data['prestige_rank_inv'])

        field_data['parleave_objective_length_women_inv'] = pd.to_numeric(
            field_data['university_name_standard'].apply(
                lambda x: parental_leave_mapping[x]['paid_leave_weeks_woman']
                if ((x in parental_leave_mapping) and
                    (parental_leave_mapping[x]['missing'] == '0'))
                else np.nan))
        field_data['parleave_objective_type_women_inv'] = field_data['university_name_standard'].apply(
            lambda x: parental_leave_mapping[x]['relief_woman']
            if ((x in parental_leave_mapping) and
                (parental_leave_mapping[x]['missing'] == '0'))
            else np.nan)

        field_data['parleave_objective_length_men_inv'] = pd.to_numeric(
            field_data['university_name_standard'].apply(
                lambda x: parental_leave_mapping[x]['paid_leave_weeks_man']
                if ((x in parental_leave_mapping) and
                    (parental_leave_mapping[x]['missing'] == '0'))
                else np.nan))
        field_data['parleave_objective_type_men_inv'] = field_data['university_name_standard'].apply(
            lambda x: parental_leave_mapping[x]['relief_man']
            if ((x in parental_leave_mapping) and
                (int(parental_leave_mapping[x]['missing']) == 0))
            else np.nan)

    all_business['likely_department'] = "Business"
    all_history['likely_department'] = "History"

    data = pd.concat([all_business, all_history], axis=0, sort=False)

    # Add a column that converts the year of birth survey categories into years
    data['age_coded'] = data['age'].apply(lambda x: codebook_age[x])
    data['isyoung'] = (2017 - data['age_coded']) < 40

    # Add a column that converts the TT start survey categories into years
    codebook_ttyear = dict(list(zip(range(1, 75, 1), range(2018, 1943, -1))))
    codebook_ttyear[-77] = None
    codebook_ttyear[0] = None

    data['ttyear_real'] = data['ttyear'].apply(lambda x: codebook_ttyear[x])

    # Separate child birth years into integer values across columns
    data['children_age'] = data['children_age'].astype(str)

    def to_numeric(x):
        try:
            num = pd.to_numeric(x)
            if num > 0:
                return num
            else:
                return None
        except ValueError:
            return None

    kids = data.children_age.str.split(
        r'\,|\;|and', expand=True).applymap(to_numeric)
    kids.columns = ['chage1', 'chage2', 'chage3', 'chage4', 'chage5', 'chage6',
                    'chage7', 'chage8']
    merged = pd.concat([data, kids], axis=1)

    # Merge productivity for faculty which we could find onto this data
    productivity = pd.read_json(BUSI_HIS_PUBS)
    productivity['pubs'] = productivity['pubs'].apply(lambda x: [[i] for i in x])

    busi_his_merged = pd.merge(merged, productivity, how='left',
                               left_on=['pid'], right_on=['sid'],
                               validate='many_to_one')

    # Rename these columns to be consistent with CS responses
    mapping = {"ttyear_real": "first_asst_job_year", "pubs": "dblp_pubs",
               "u_name": "name", "gender": "gender_ans"}
    busi_his_merged.rename(columns=mapping, inplace=True)

    return busi_his_merged


def load_cs_faculty():
    frame = pd.read_excel(CS_FRAME)

    # Let's recalculate these variables:
    frame.drop(['prestige', 'parleave_objective_length_women',
                'parleave_objective_type_women',
                'parleave_objective_length_men',
                'parleave_objective_type_men'], axis=1, inplace=True)

    # Standardize university names (just a big lookup table)
    frame['university_name_standard'] = frame['university'].apply(lambda x: INST_NAME_ALIASES[x] if x in INST_NAME_ALIASES else x)

    pi_rank_mapping = {}
    parental_leave_mapping = {}

    # Extract prestige / ranking data and parental leave information
    with open(PRESTIGE % 'CS') as rankings, \
         open(PARENTAL_LEAVE) as parental_leave:
        pi_reader = csv.DictReader(rankings, dialect='excel-tab')
        for row in pi_reader:
            pi_rank_mapping[row['institution']] = row

        leave_reader = csv.DictReader(parental_leave, dialect='excel-tab')
        for row in leave_reader:
            parental_leave_mapping[row['university_name']] = row

    frame['prestige_inv'] = frame['university_name_standard'].apply(lambda x: pi_rank_mapping[x]['pi'] if x in pi_rank_mapping else np.nan)
    frame['prestige_rank_inv'] = frame['university_name_standard'].apply(lambda x: pi_rank_mapping[x]['# u'] if x in pi_rank_mapping else np.nan)

    frame['prestige_inv'] = pd.to_numeric(frame['prestige_inv'])
    frame['prestige_rank_inv'] = pd.to_numeric(frame['prestige_rank_inv'])

    frame['parleave_objective_length_women'] = pd.to_numeric(frame['university_name_standard'].apply(
        lambda x: parental_leave_mapping[x]['paid_leave_weeks_woman']
        if ((x in parental_leave_mapping) and
            (parental_leave_mapping[x]['missing'] == '0')) else np.nan))
    frame['parleave_objective_type_women'] = frame['university_name_standard'].apply(
        lambda x: parental_leave_mapping[x]['relief_woman']
        if ((x in parental_leave_mapping) and
            (parental_leave_mapping[x]['missing'] == '0')) else np.nan)

    frame['parleave_objective_length_men'] = pd.to_numeric(frame['university_name_standard'].apply(
        lambda x: parental_leave_mapping[x]['paid_leave_weeks_man']
        if ((x in parental_leave_mapping) and
            (parental_leave_mapping[x]['missing'] == '0')) else np.nan))
    frame['parleave_objective_type_men'] = frame['university_name_standard'].apply(
        lambda x: parental_leave_mapping[x]['relief_man'] if ((x in parental_leave_mapping) and (int(parental_leave_mapping[x]['missing']) == 0)) else np.nan)

    # Merge productivity
    productivity = pd.read_json(CS_PUBS)
    cs_ans = pd.read_excel(CS_RESPONSES)
    cs_inv = pd.merge(frame, productivity, how='left', left_on='dblp',
                      right_on='dblp', validate='many_to_one',
                      suffixes=['', '_prod'])

    cs_merged = cs_ans.merge(cs_inv, left_on='sid', right_on='survey sid',
                             suffixes=('_ans', '_inv'))

    cs_merged['isyoung'] = cs_merged['age (actual)'] < 40
    cs_merged['likely_department'] = "Computer Science"

    cs_merged['age_coded'] = cs_merged['age (coded)'].apply(
        lambda x: codebook_age[x] if x in codebook_age else None)

    return cs_merged
