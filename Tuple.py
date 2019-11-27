'''
object:

'''
import csv

path = './data'
file = '/example-dirty.csv'

delta = 3
tuples = list()

dict_predict = dict()
dict_predict_re = dict()

'''
Tuple Class
cid --- the only id can identify the tuple
value_dict --- attribute: value (e.g. {'CC': '01', 'AC': '908', 'ZIT': '07974', ...})
feature_vec --- [0, 1, 0, ...]
label --- 1 or 0
'''
class Tuple():
    def __init__(self):
        self.cid = -1
        self.value_dict = dict()
        self.feature_vec = list()
        self.label = None

# '''
# # print the attributes and values
# with open(path+file, encoding='utf-8') as csvfile:
#     reader = csv.reader(csvfile)
#     for row in reader:
#         print(row)
# '''

'''
# create two dictionaries
# attr_dict is {index: attribute, ...} (e.g. {6: 'ZIP', ...})
# attr_dict_re is {attribute: index, ...} (e.g. {'ZIP': 6, ...})
'''
with open(path+file, encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    attr_name = next(reader)
    attr_dict = {i: attr for i, attr in enumerate(attr_name)}
    attr_dict_re = {attr: i for i, attr in enumerate(attr_name)}
    # print(attr_dict)
    # print(attr_dict_re)

'''
tuples:
a list of Tuple Class
'''
with open(path+file, encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # print(row)
        tup = Tuple()
        tup.value_dict = {value: row[value] for i, value in attr_dict.items()}
        tuples.append(tup)

out_path = './output'
out_file = '/output.txt'

with open(out_path+out_file, mode='w', encoding='utf-8') as writer:
    for tuple in tuples:
        for value in tuple.value_dict.values():
            writer.write(value + ' ')
        writer.write('\n')

'''
dict_freq
{'ZIP': {'07974': 4, '07975': 5, ...}, 'CT': {'MH': 2, ...}, ...}
'''
dict_freq = dict()
dict_freq_dict = dict()
for tuple in tuples:
    for key, value in tuple.value_dict.items():

        if key not in dict_freq.keys():
            dict_freq[key] = dict()
        dict_freq_dict = dict_freq[key]

        if value not in dict_freq_dict:
            dict_freq_dict[value] = 0
        dict_freq_dict[value] += 1

        dict_freq[key] = dict_freq_dict

print(dict_freq)

'''
delta = 2
predict can be selected if its frequent >= 2
'''
index = 0
for attr, dict_freq_dict in dict_freq.items():
    for key, value in dict_freq_dict.items():
        if value >= delta:
            str_predict = str(attr) + '=' + str(key)
            dict_predict[index] = str_predict
            dict_predict_re[str_predict] = index
            index += 1

print(dict_predict)
print(dict_predict_re)

'''
generate the feature vector of every tuple
'''
for tup in tuples:
    for i in dict_predict.keys():
        tup.feature_vec.append(0)

    for attr, value in tup.value_dict.items():
        str_predict = str(attr) + '=' + str(value)
        if str_predict in dict_predict_re.keys():
            tup.feature_vec[dict_predict_re[str_predict]] = 1

'''
generate the cid of every tuple
'''
for index, tup in enumerate(tuples):
    tup.cid = index

# input_str = input("Which tuple does violate the CFDs you want to express? (Please input the cid of tuples, e.g. 2,3,4,5) >>> ")
input_str = '2,3,4,5'

input_cid = list(map(int, input_str.split(',')))
for cid in input_cid:
    tuples[cid].label = 1

for tup in tuples:
    print(tup.cid, tup.value_dict, tup.feature_vec, tup.label)