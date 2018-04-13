import csv


def read_documents(dataset_file, has_header):
    with open(dataset_file, 'r', encoding="utf8") as dataset:
        tsv_dataset_reader = csv.reader(dataset, delimiter='\t', )
        if has_header:
            next(tsv_dataset_reader)
        for row in tsv_dataset_reader:
            yield row
            # yield [unicode(cell, 'utf-8') for cell in row]


def file_len(file_name):
    i = 0
    with open(file_name, encoding="utf8") as f:
        for i, l in enumerate(f):
            pass
    return i + 1
