
import csv
import argparse

from genderize import Genderize

parser = argparse.ArgumentParser()
parser.add_argument("filename")
args = parser.parse_args()
print(args.filename)  # Must be a TSV

first_names = []
with open(args.filename) as tsvfile:
    reader = csv.DictReader(tsvfile)
    for row in reader:
        if ('firstname' in row) and len(row['firstname']) > 0 and \
         not any(char.isdigit() for char in row['firstname']):
            first_name = row['firstname']
        else:
            if 'u_name' in row:
                first_name = row['u_name'].split(' ')[0]
            if 'name' in row:
                first_name = row['name'].split(' ')[0]

    first_names.append(first_name)

first_names_set = {}
output = args.filename.rstrip(r'\.tsv') + '_genderize.tsv'

with open(output, 'w') as tsvfile:
    writer = csv.writer(tsvfile, delimiter='\t')
    for i, first_name in enumerate(first_names):
        print(i, first_name)
        if first_name in first_names_set:
            writer.writerow([first_names_set[first_name]])
            continue

        res = Genderize().get([first_name])[0]
        if res['probability'] > 0.95:
            first_names_set[first_name] = res['gender']
            writer.writerow([res['gender']])
        else:
            first_names_set[first_name] = 'unclear'
            writer.writerow(['unclear'])
