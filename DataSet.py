import csv
import os

path = './data'
file = '/example-cleaned.csv'

with open(path+file, encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    reader = next(reader)
    attr_dict = {i: attr for i, attr in enumerate(reader)}
    attr_dict_re = {attr: i for i, attr in enumerate(reader)}

print(attr_dict)
print(attr_dict_re)


