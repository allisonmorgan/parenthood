
import pandas as pd

from parse.mturk import parse_faculty_from_link_file
from parse.institution_parser import parse_institution_records
from parse.load import load_all_publications, load_assistant_profs

"""
Script which takes all of the DBLP profiles downloaded and converts them into
a nice JSON file for merging onto responses.
"""

# Load new faculty publication data
faculty = []
faculty_file = '../../data/survey_data/cs_survey/mturk_DBLP_02-09-18.xlsx'
inst_file = '../../data/survey_data/faculty_2011/inst_cs_CURRENT.txt'
# DBLP profiles recollected Feb 2020
dblp_dir = '../../data/survey_data/new_dblp_records_processed'

inst = parse_institution_records(open(inst_file, 'r'))
new_faculty = parse_faculty_from_link_file(faculty_file)
load_all_publications(new_faculty, dblp_dir)
faculty += new_faculty

# Load earlier faculty and their publication data
faculty_file = '../../data/survey_data/faculty_2011/faculty_cs_CURRENT.txt'
inst_file = '../../data/survey_data/faculty_2011/inst_cs_CURRENT.txt'

inst = parse_institution_records(open(inst_file, 'r'))
earlier_faculty = load_assistant_profs(open(faculty_file, 'r',
                                            encoding='windows-1252'), inst,
                                       ranking='pi')
load_all_publications(earlier_faculty, dblp_dir)
faculty += earlier_faculty

df = pd.DataFrame()

dblp = []
pub_timeseries = []
name = []
tt_start_dates = []
places = []
emails = []
gss = []

for f in faculty:
    if 'facultyName' in f:
        tt_start_date = None
        dblp_id = None
        pub_series = None
        facultyName = None
        place = None
        email = None
        gs = None
        pubs = None

        facultyName = f['facultyName']

        if 'dblp' in f:
            dblp_id = f['dblp']
            pubs = f['dblp_pubs']

            # Filter by publication type
            pub_types = ['article', 'inproceedings']
            pub_series = [[pub['year'], pub['authors']]
                          for pub in pubs if pub['pub_type'] in pub_types]
            # pub_series = [[pub] for pub in pubs]

        if 'first_asst_job_year' in f:
            tt_start_date = f['first_asst_job_year']

        if 'place' in f:
            place = f['place'].encode(encoding='utf-8')

        if 'email' in f:
            email = f['email'].encode(encoding='utf-8').lower()

        if 'gs' in f:
            gs = f['gs'].encode(encoding='utf-8')

        emails.append(email)
        places.append(place)
        name.append(facultyName)
        dblp.append(dblp_id)
        pub_timeseries.append(pub_series)
        tt_start_dates.append(tt_start_date)
        gss.append(gs)

df['facultyName'] = name
df['dblp'] = dblp
df['dblp_pubs'] = pub_timeseries
df['place'] = places
df['email'] = emails
df['google_scholar'] = gss
df['first_asst_job_year'] = tt_start_dates

df.drop_duplicates(subset=['dblp'], inplace=True)
df.to_json(
    '../../data/survey_data/cs_pubs/cs_prior_productivity_authorship_feb5_2020.json',
    orient='records')
