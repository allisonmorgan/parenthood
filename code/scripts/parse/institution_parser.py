#!/usr/bin/env python
"""
Parsing of university record files.
"""

import numpy as np

def custom_cast(x, cast_types=[int, float, str]):
    """ Attempt to cast x using the specified types in the order
        in which they appear.  """ 
    for cast_func in cast_types:
        try:
            return cast_func(x)
        except ValueError:
            pass
    raise BaseException('All casts failed!')

def parse_institution_records(fp):
    """ Parse a university record file.
        Inputs:
          + fp - an *open* file pointer containing
                 university/institution records.

        Yields:
          + A dictionary linking an institution name
            to the attributes stored in the file.

        Example:
            >>> institutions = parse_institution_records(X)
            >>> print institutions['Carnegie Mellon University'].USN2010
                1
            >>> print institutions['Harvard University'].pi
                6.12
            >>> print institutions['Yale University'].Region
                'Northeast'

        NOTE:  Attributes are just the column names in the
               institution records file.
    """
    institutions = {}
    detect_header = True
    for line in fp:
        line = line.strip()
        if not line:
            continue  # Skip empty lines

        if detect_header:
            detect_header = False
            if not line.startswith('# '):
                raise ValueError('File does not appear to be a valid '
                                 'institution records file!')
            line = line[2:]  # Remove leading "# "
            fields = line.split('\t')
            if 'institution' not in fields:
                raise ValueError('Records file missing `institution` field!')
            num_fields = len(fields)
            institution_field_ind = fields.index('institution')
            info_fields = [f.strip() for i, f in enumerate(fields)
                           if i != institution_field_ind]
            info_fields_ind = [i for i in range(num_fields)
                               if i != institution_field_ind]
        else:
            fields = line.strip().split('\t')
            if len(fields) != num_fields:
                raise ValueError('Missing/extra fields in line: %s' % line)
            info_values = [custom_cast(fields[i].strip())
                           for i in info_fields_ind]
            institution = fields[institution_field_ind].strip()
            institution_record = dict(zip(info_fields, info_values))
            institutions[institution] = institution_record

    # Fill in variations on `pi'
    ranks = []
    for i in institutions:
        temp = institutions[i].get('pi', np.inf)
        if temp < np.inf:
            ranks.append(temp)
    ranks = np.array(ranks)

    ranks.sort()
    delta = np.mean(ranks[1:] - ranks[0:-1])
    worst_ranking = ranks.max()
    best_ranking = ranks.min()
    scale = lambda x: 1. - (x-best_ranking)/(worst_ranking-best_ranking+delta)

    '''
    VARIATION #1: `pi_inv' - Inverse of the pi ranking. Lower ranks
    yield larger numbers.
    '''
    for i in institutions:
        institutions[i]['pi_inv'] = 1. / institutions[i].get('pi',
                                                             worst_ranking)

    '''
    VARIATION #2: `pi_rescaled' - Rescaled such that the best school gets 1.0
                                      and the rest get some epsilon value.
    '''
    for i in institutions:
        institutions[i]['pi_rescaled'] = scale(
            institutions[i].get('pi', worst_ranking))

    # Fill in defaults
    institutions['UNKNOWN'] = {}
    institutions['UNKNOWN']['pi'] = worst_ranking
    institutions['UNKNOWN']['pi_inv'] = 1./worst_ranking
    institutions['UNKNOWN']['pi_rescaled'] = scale(worst_ranking)
    institutions['UNKNOWN']['Region'] = 'Earth'

    return institutions


# Discrepancies between the frame files and 2011 data in
# terms of institution names
INST_NAME_ALIASES = {}
for alias in ['CU Boulder', 'University of Colorado Boulder',
              'university of colorado', 'University of Colorado',
              'university of colorado boulder',
              'University of Colorado-Boulder',
              'University of Colorado at Boulder', 'Leeds School of Business']:
    INST_NAME_ALIASES[alias] = 'University of Colorado, Boulder'
for alias in ['Binghamton University, State University of New York',
              'Binghamton University, SUNY']:
    INST_NAME_ALIASES[alias] = 'State University of New York, Binghamton'
for alias in ['Calgary']:
    INST_NAME_ALIASES[alias] = 'University of Calgary'
for alias in ['University of Hawaii', 'University of Hawaii at Manoa']:
    INST_NAME_ALIASES[alias] = 'University of Hawaii, Manoa'
for alias in ['Oregon Health & Science University', 'OHSU']:
    INST_NAME_ALIASES[alias] = 'Oregon Health and Science University'
for alias in ['Massachusetts Institute of Technology']:
    INST_NAME_ALIASES[alias] = 'MIT'
for alias in ['Columbia']:
    INST_NAME_ALIASES[alias] = 'Columbia University'
for alias in ['Yale']:
    INST_NAME_ALIASES[alias] = 'Yale University'
for alias in ['University of California, Berkeley']:
    INST_NAME_ALIASES[alias] = 'UC Berkeley'
for alias in ['University of California, Davis']:
    INST_NAME_ALIASES[alias] = 'UC Davis'
for alias in ['University of California, Irvine']:
    INST_NAME_ALIASES[alias] = 'UC Irvine'
for alias in ['University of California, Los Angeles',
              'University of California--Los Angeles']:
    INST_NAME_ALIASES[alias] = 'UCLA'
for alias in ['University of California, Riverside']:
    INST_NAME_ALIASES[alias] = 'UC Riverside'
for alias in ['University of California, San Diego', 'UC, San Diego']:
    INST_NAME_ALIASES[alias] = 'UC San Diego'
for alias in ['University of California, Santa Barbara']:
    INST_NAME_ALIASES[alias] = 'UC Santa Barbara'
for alias in ['University of California, Santa Cruz']:
    INST_NAME_ALIASES[alias] = 'UC Santa Cruz'
for alias in ['university of florida', 'U of Florida', 'Univ. of Florida',
              'UF', 'University of Florida, Gainesville']:
    INST_NAME_ALIASES[alias] = 'University of Florida'
for alias in ['Georgia Institute of Technology']:
    INST_NAME_ALIASES[alias] = 'Georgia Tech'
for alias in ['City University of New York, Graduate Center']:
    INST_NAME_ALIASES[alias] = 'CUNY Graduate Center'
for alias in ["Queen's University", "Queen's University at Kingston",
              "Queen's University, Kingston Ontario"]:
    INST_NAME_ALIASES[alias] = 'Queens University'
for alias in ['College of William & Mary']:
    INST_NAME_ALIASES[alias] = 'College of William and Mary'
for alias in ['Oakland University']:
    INST_NAME_ALIASES[alias] = 'Oakland University (Michigan)'
for alias in ['Waterloo', 'Univerisity of Waterloo']:
    INST_NAME_ALIASES[alias] = 'University of Waterloo'
for alias in ['Toyota Technological Institute, Chicago']:
    INST_NAME_ALIASES[alias] = 'Toyota Technological Institute at Chicago'
for alias in ['Harvard']:
    INST_NAME_ALIASES[alias] = 'Harvard University'
for alias in ['Lehigh']:
    INST_NAME_ALIASES[alias] = 'Lehigh University'
for alias in ['Clemson']:
    INST_NAME_ALIASES[alias] = 'Clemson University'
for alias in ['University of Illinois at Urbana-Champaign',
              'University of Illinois', 'UIUC',
              'University of Illinois at Urbana Champaign',
              'University of Illinois, Urbana-Champaign']:
    INST_NAME_ALIASES[alias] = 'University of Illinois, Urbana Champaign'
for alias in ['University at Albany SUNY', 'University at Albany',
              'University of Albany, SUNY', 'University at Albany, SUNY']:
    INST_NAME_ALIASES[alias] = 'State University of New York, Albany'
for alias in ['University of Buffalo, SUNY',
              'University at Buffalo, The State University of New York',
              'University at Buffalo', 'University at Buffalo, SUNY']:
    INST_NAME_ALIASES[alias] = 'State University of New York, Buffalo'
for alias in ['depaul university']:
    INST_NAME_ALIASES[alias] = 'DePaul University'
for alias in ['The University of Texas at Austin', 'UT Austin',
              'University of Texas at Austin']:
    INST_NAME_ALIASES[alias] = 'University of Texas, Austin'
for alias in ['Oklahoma State']:
    INST_NAME_ALIASES[alias] = 'Oklahoma State University'
for alias in ['old dominion university']:
    INST_NAME_ALIASES[alias] = 'Old Dominion University'
for alias in ['Concordia University']:
    INST_NAME_ALIASES[alias] = 'Concordia University, Montreal'
for alias in ['Georgia State']:
    INST_NAME_ALIASES[alias] = 'Georgia State University'
for alias in ['University of Idaho']:
    INST_NAME_ALIASES[alias] = 'University of Idaho, Moscow'
for alias in ['Texas A&M University', 'Texas A&M University, College Station']:
    INST_NAME_ALIASES[alias] = 'Texas A&M'
for alias in ['Stony Brook University, SUNY']:
    INST_NAME_ALIASES[alias] = 'State University of New York, Stony Brook'
for alias in ['New Mexico Tech']:
    INST_NAME_ALIASES[alias] = 'New Mexico Institute of Mining and Technology'
for alias in ['Missouri University of Science & Technology']:
    INST_NAME_ALIASES[alias] = 'Missouri University of Science and Technology'
for alias in ['University of Illinois at Chicago',
              'University of Illinois--Chicago']:
    INST_NAME_ALIASES[alias] = 'University of Illinois, Chicago'
for alias in ['University of Louisiana Lafayette']:
    INST_NAME_ALIASES[alias] = 'University of Louisiana, Lafayette'
for alias in ['Polytechnic Institute of New York University']:
    INST_NAME_ALIASES[alias] = 'Polytechnic Institute of NYU'
for alias in ['UNC Charlotte', 'University of North Carolina at Charlotte']:
    INST_NAME_ALIASES[alias] = 'University of North Carolina, Charlotte'
for alias in ['The Ohio State University', 'Ohio State']:
    INST_NAME_ALIASES[alias] = 'Ohio State University'
for alias in ['Univ of Chicago']:
    INST_NAME_ALIASES[alias] = 'University of Chicago'
for alias in ['University of Texas at San Antonio']:
    INST_NAME_ALIASES[alias] = 'University of Texas, San Antonio'
for alias in ['University of Wisconsin-Milwaukee']:
    INST_NAME_ALIASES[alias] = 'University of Wisconsin, Milwaukee'
for alias in ['Penn State University']:
    INST_NAME_ALIASES[alias] = 'Pennsylvania State University'
for alias in ['Southern Illinois university']:
    INST_NAME_ALIASES[alias] = 'Southern Illinois University, Carbondale'
for alias in ['University of Kansas', 'Univ of Kansas']:
    INST_NAME_ALIASES[alias] = 'University of Kansas, Lawrence'
for alias in ['University of Maryland', 'Maryland', 'UMD']:
    INST_NAME_ALIASES[alias] = 'University of Maryland, College Park'
for alias in ['University of Minnesota', 'U Minnesota', 'Minnesota',
              'University of Minnesota, Twin Cities']:
    INST_NAME_ALIASES[alias] = 'University of Minnesota, Minneapolis'
for alias in ['University of Nebraska-Lincoln']:
    INST_NAME_ALIASES[alias] = 'University of Nebraska, Lincoln'
for alias in ['University of North Texas']:
    INST_NAME_ALIASES[alias] = 'University of North Texas, Denton'
for alias in ['GMU']:
    INST_NAME_ALIASES[alias] = 'George Mason University'
for alias in ['Washington State University']:
    INST_NAME_ALIASES[alias] = 'Washington State University, Pullman'
for alias in ['University of Alabama', 'Univ of Alabama', 'U. of Alabama']:
    INST_NAME_ALIASES[alias] = 'University of Alabama, Tuscaloosa'
for alias in ['UMass Lowell']:
    INST_NAME_ALIASES[alias] = 'University of Massachusetts, Lowell'
for alias in ['U of Denver']:
    INST_NAME_ALIASES[alias] = 'University of Denver'
for alias in ['Universitiy of Lousville']:
    INST_NAME_ALIASES[alias] = 'University of Louisville'
for alias in ['U. of Arkansas at Little Rock']:
    INST_NAME_ALIASES[alias] = 'University of Arkansas, Little Rock'
for alias in ['UCCS']:
    INST_NAME_ALIASES[alias] = 'University of Colorado, Colorado Springs'
for alias in ['Universit√© de Montr√©al']:
    INST_NAME_ALIASES[alias] = 'University of Montreal'
for alias in ['Miami University']:
    INST_NAME_ALIASES[alias] = 'Miami University, Ohio'
for alias in ['University of Missouri - Columbia', 'University of Missouri']:
    INST_NAME_ALIASES[alias] = 'University of Missouri, Columbia'
for alias in ['University of Arkansas']:
    INST_NAME_ALIASES[alias] = 'University of Arkansas, Fayetteville'
for alias in ['Univ of Mississippi']:
    INST_NAME_ALIASES[alias] = 'University of Mississippi'
for alias in ['Umiversity of Michigan', 'University of Michigan, Ann Arbor']:
    INST_NAME_ALIASES[alias] = 'University of Michigan'
for alias in ['U Western Michigan', 'Western Michigan Univ.',
              'Western Michigan Unversity']:
    INST_NAME_ALIASES[alias] = 'Western Michigan University'
for alias in ['U of Pennsylvania',
              'Wharton School, University of Pennsylvania']:
    INST_NAME_ALIASES[alias] = 'University of Pennsylvania'
for alias in ['CMU', 'Carnegie Mellon', 'Carnegie mellon']:
    INST_NAME_ALIASES[alias] = 'Carnegie Mellon University'
for alias in ['University of North Carolina at Chapel Hill',
              'UNC Chapel Hill', 'North Carolina', 'UNC']:
    INST_NAME_ALIASES[alias] = 'University of North Carolina, Chapel Hill'
for alias in ['Babson college']:
    INST_NAME_ALIASES[alias] = 'Babson College'
for alias in ['The University of Texas at Arlington',
              'University of Texas at Arlington']:
    INST_NAME_ALIASES[alias] = 'University of Texas, Arlington'
for alias in ['Stanford']:
    INST_NAME_ALIASES[alias] = 'Stanford University'
for alias in ['Indiana University-Bloomington',
              'Indiana University, Bloomington']:
    INST_NAME_ALIASES[alias] = 'Indiana University'
for alias in ['Case Western reserve university']:
    INST_NAME_ALIASES[alias] = 'Case Western Reserve University'
for alias in ['Iowa State Univ']:
    INST_NAME_ALIASES[alias] = 'Iowa State University'
for alias in ['Rutgers, the State University of New Jersey',
              'RUTGERS UNIVERSITY', 'Rutgers',
              'Rutgers University - New Brunswick']:
    INST_NAME_ALIASES[alias] = 'Rutgers University'
for alias in ['university of virginia', 'Univ. of Virginia']:
    INST_NAME_ALIASES[alias] = 'University of Virginia'
for alias in ['Kent State University Stark', 'Kent State University- Kent']:
    INST_NAME_ALIASES[alias] = 'Kent State University'
for alias in ['Arizona State U']:
    INST_NAME_ALIASES[alias] = 'Arizona State University'
for alias in ['southern methodist university']:
    INST_NAME_ALIASES[alias] = 'Southern Methodist University'
for alias in ['BYU']:
    INST_NAME_ALIASES[alias] = 'Brigham Young University'
for alias in ['NYU']:
    INST_NAME_ALIASES[alias] = 'New York University'
for alias in ['The Catholic University of America', 'Catholic University']:
    INST_NAME_ALIASES[alias] = 'Catholic University of America'
for alias in ['CUNY Queens College', 'Queens College, CUNY',
              'Queens College - CUNY']:
    INST_NAME_ALIASES[alias] = 'CUNY Graduate Center'
for alias in ['Oklahoma']:
    INST_NAME_ALIASES[alias] = 'University of Oklahoma'
for alias in ['Auburn U.']:
    INST_NAME_ALIASES[alias] = 'Auburn University'
for alias in ['UConn']:
    INST_NAME_ALIASES[alias] = 'University of Connecticut'
for alias in ['Marquette U', 'Marquette']:
    INST_NAME_ALIASES[alias] = 'Marquette University'
for alias in ['Univ of Arizona']:
    INST_NAME_ALIASES[alias] = 'University of Arizona'
for alias in ['Purdue', 'Purdue University, West Lafayette']:
    INST_NAME_ALIASES[alias] = 'Purdue University'
for alias in ['Syracuse']:
    INST_NAME_ALIASES[alias] = 'Syracuse University'
for alias in ['northwestern']:
    INST_NAME_ALIASES[alias] = 'Northwestern University'
for alias in ['Vanderbilt U']:
    INST_NAME_ALIASES[alias] = 'Vanderbilt University'
for alias in ['UGA']:
    INST_NAME_ALIASES[alias] = 'University of Georgia'
for alias in ['Baruch']:
    INST_NAME_ALIASES[alias] = 'CUNY Baruch College'
for alias in ['Univ. of New Mexico']:
    INST_NAME_ALIASES[alias] = 'University of New Mexico'
for alias in ['Rice']:
    INST_NAME_ALIASES[alias] = 'Rice University'
for alias in ['Utah']:
    INST_NAME_ALIASES[alias] = 'University of Utah'
for alias in ['Rice']:
    INST_NAME_ALIASES[alias] = 'Rice University'
for alias in ['Baruch', 'Bernard M. Baruch College',
              'City University of New York, Bernard M. Baruch College']:
    INST_NAME_ALIASES[alias] = 'CUNY Baruch College'
for alias in ['Louisiana State University, Baton Rouge']:
    INST_NAME_ALIASES[alias] = 'Louisiana State University'
for alias in ['St. Louis University']:
    INST_NAME_ALIASES[alias] = 'Saint Louis University'
for alias in ['University of Wisconsin--Madison']:
    INST_NAME_ALIASES[alias] = 'University of Wisconsin, Madison'
for alias in ['University of Miami, Coral Gables']:
    INST_NAME_ALIASES[alias] = 'University of Miami'
for alias in ['American University']:
    INST_NAME_ALIASES[alias] = 'American University, Washington'
for alias in ['Washington University in St. Louis']:
    INST_NAME_ALIASES[alias] = 'Washington University, St. Louis'
for alias in ['Graduate Theological Union, Berkeley']:
    INST_NAME_ALIASES[alias] = 'Graduate Theological Union'
for alias in ["St. John's University, Queens"]:
    INST_NAME_ALIASES[alias] = 'Saint Johns University'
for alias in ["Florida State University, Tallahassee"]:
    INST_NAME_ALIASES[alias] = 'Florida State University'
