
import os
import re
import pickle
import urllib.request

from parse.mturk import parse_faculty_from_link_file
from parse.load import load_all_profs
from parse.institution_parser import parse_institution_records
from parse.dblp import parse_dblp_publications

"""
Script which downloads all the the DBLP profiles of the 2011/2017 faculty.
Modified from:
https://github.com/samfway/faculty_hiring/blob/master/scripts/download_all.py
"""


def get_author_tag(dblp_url):
    return re.findall(r'(?<=pers/hd/./)[^,]+(?=,?)', dblp_url)[0]


def get_dblp_url(author_tag):
    first_letter = author_tag[0].lower()
    return('http://dblp.uni-trier.de/pers/hd/%s/%s' %
           (first_letter, author_tag))


def download_dblp_page(dblp_url, output_prefix, page_number=0):
    author_tag = get_author_tag(dblp_url)
    filename = output_prefix + author_tag + '_file_%d.html' % (page_number)
    fname, response = urllib.request.urlretrieve(dblp_url, filename)
    return response, filename


# Create locations for DBLP data to be stored
dblp_dir = "../../data/survey_data/new_dblp_records_html"
dblp_prefix = os.path.join(dblp_dir, 'DBLP_')
inst_file = '../../data/survey_data/faculty_2011/inst_cs_CURRENT.txt'

# File locations
faculty_file = '../../data/survey_data/mturk_DBLP_02-09-18.xlsx'
faculty = []
inst = parse_institution_records(open(inst_file, 'r'))
faculty += parse_faculty_from_link_file(faculty_file)

# Load earlier faculty
faculty_file = '../../data/survey_data/faculty_2011/faculty_cs_CURRENT.txt'
earlier_faculty = load_all_profs(
    open(faculty_file, 'r', encoding='windows-1252'), inst, ranking='pi')
faculty += earlier_faculty

DBLP_FILE = 'DBLP_%s_file_0.html'

# For each person in faculty, get their DBLP profile
for i, person in enumerate(faculty):
    if 'dblp' in person:
        dblp_file = os.path.join(dblp_dir, DBLP_FILE % person['dblp'])
        if not os.path.isfile(dblp_file):
            print('%.3f' % (i/len(faculty)),
                  'DBLP -> ', person['facultyName'],
                  '[%s]' % person['dblp'])
            dblp_url = get_dblp_url(person['dblp'])
            download_dblp_page(dblp_url, dblp_prefix)

# Parse DBLP publication pages
parse_dblp_publications(faculty, dblp_dir)
DBLP_PKL = 'DBLP_%s.pkl'
processed_dir = '../../data/survey_data/new_dblp_records_processed/'

# For each faculty member, save a pickle file with their publication data
for person in faculty:
    if 'dblp_pubs' in person:
        output_file = os.path.join(processed_dir, DBLP_PKL % person['dblp'])
        if not os.path.isfile(output_file):
            with open(output_file, 'wb') as fp:
                pickle.dump(person['dblp_pubs'], fp)
                pickle.dump(person['dblp_stats'], fp)
