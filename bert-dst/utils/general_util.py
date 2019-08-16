import csv


def read_tsv(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
    lines = []
    for line in reader:
        if len(line) > 0 and line[0][0] == '#':  # Remove comment
            lines.append(line)
    return lines
