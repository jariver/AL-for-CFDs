import csv
import os

path = './data'
file = '/example-cleaned.csv'

tuples = list()

with open(path+file, encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        print(row)

with open(path+file, encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    attr_name = next(reader)
    attr_dict = {i: attr for i, attr in enumerate(attr_name)}
    attr_dict_re = {attr: i for i, attr in enumerate(attr_name)}

with open(path+file, encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # print(row)
        tuples.append({value: row[value] for i, value in attr_dict.items()})

print(tuples)

out_path = './output'
out_file = '/output.txt'

with open(out_path+out_file, mode='w', encoding='utf-8') as writer:
    for tuple in tuples:
        for value in tuple.values():
            writer.write(value + ' ')
        writer.write('\n')

