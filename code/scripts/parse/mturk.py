import pandas as pd
import re


def parse_faculty_from_link_file(dblp_link_file):
    """ Parse partial faculty records out of our MTurk file. """
    parsed_faculty = []
    link_df = pd.read_excel(dblp_link_file)

    for _, row in link_df.iterrows():
        person = {}
        person['facultyName'] = row['Input.faculty_name']
        person['place'] = row['Input.institution']
        person['first_asst_job_year'] = row['Answer.current_start_date']
        person['phd_year'] = row['Answer.phd_date']

        # If the person has a legit-looking DBLP record, grab it.
        if 'pers/hd' in row['Answer.dblp_url']:
            dblp_id = re.findall('(?<=hd/./).*', row['Answer.dblp_url'])[0]
            person['dblp'] = dblp_id
            person['dblp_url'] = row['Answer.dblp_url']

        # gs = "https://scholar.google.com/citations?user="
        if 'scholar.google.com' and 'user=' in row['Answer.google_url']:
            tmp = re.split(r'(\?|&)', row['Answer.google_url'])
            gs_id = [part.replace("user=", "")
                     for part in tmp if part.count("user=")][0]
            person['gs'] = gs_id
            person['gs_url'] = row['Answer.google_url']

        # Grab their rank
        if row['Answer.rank'] == 'assistant_professor':
            person['rank'] = 'Assistant Professor'
        elif row['Answer.rank'] == 'associate_professor':
            person['rank'] = 'Associate Professor'
        elif row['Answer.rank'] == 'full_professor':
            person['rank'] = 'Full Professor'
        else:
            person['rank'] = None

        # Add record to the list
        if person['rank']:
            parsed_faculty.append(person.copy())

    return parsed_faculty
