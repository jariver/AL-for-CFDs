import csv
from Tuple import *

def create_attr_dict(path, file):
    r'''Create two dictionaries using to mark the index of Attributes.

    Args:
        path: Dictionary of the csv file.
        file: The name of the csv file.

    Return:
        attr_dict: {index: attribute, ...}, e.g.{6: 'ZIP', ...}.
        attr_dict_re: {attribute: index, ...}, e.g.{'ZIP': 6, ...}.

    '''
    with open(path+file, encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        attr_name = next(reader)
        attr_dict = {i: attr for i, attr in enumerate(attr_name)}
        attr_dict_re = {attr: i for i, attr in enumerate(attr_name)}
    return attr_dict, attr_dict_re

def create_tuples(path, file, attr_dict):
    r'''Create a list of Tuple Class.

    Args:
        path: Dictionary of the csv file.
        file: The name of the csv file.
        attr_dict: {index: attribute, ...}, e.g.{6: 'ZIP', ...}.

    Return:
        tuples: A list of all instances of Tuple Class, which contains all tuples in whole csv file.
    '''
    tuples = list()
    with open(path+file, encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            tup = Tuple()
            tup.value_dict = {value: row[value] for i, value in attr_dict.items()}
            tuples.append(tup)
    return tuples

def write_data(out_path, out_file, tuples):
    with open(out_path+out_file, mode='w', encoding='utf-8') as writer:
        for tuple in tuples:
            for value in tuple.value_dict.values():
                writer.write(value + ' ')
            writer.write('\n')

def compute_freq(tuples):
    r'''Compute the frequency of all values.

    Args:
        tuples: A list of all instances of Tuple Class, which contains all tuples in whole csv file.

    Return:
        dict_freq: {'ZIP': {'07974': 4, '07975': 5, ...}, 'CT': {'MH': 2, ...}, ...}
    '''
    dict_freq = dict()

    for tuple in tuples:
        for key, value in tuple.value_dict.items():

            if key not in dict_freq.keys():
                dict_freq[key] = dict()
            dict_freq_dict = dict_freq[key]

            if value not in dict_freq_dict:
                dict_freq_dict[value] = 0
            dict_freq_dict[value] += 1

            dict_freq[key] = dict_freq_dict
    return dict_freq

def compute_predict(dict_freq, delta):
    r'''
    delta = 2
    predict can be selected if its frequent >= 2
    '''
    dict_predict = dict()
    dict_predict_re = dict()

    index = 0
    for attr, dict_freq_dict in dict_freq.items():
        for key, value in dict_freq_dict.items():
            if value >= delta:
                str_predict = str(attr) + '=' + str(key)
                dict_predict[index] = str_predict
                dict_predict_re[str_predict] = index
                index += 1
    return dict_predict, dict_predict_re

def compute_feat_vec(tuples, dict_predict, dict_predict_re):
    r'''Compute the feature vector of every tuple and generate the cid.
    '''
    for tup in tuples:
        for i in dict_predict.keys():
            tup.feature_vec.append(0)

        for attr, value in tup.value_dict.items():
            str_predict = str(attr) + '=' + str(value)
            if str_predict in dict_predict_re.keys():
                tup.feature_vec[dict_predict_re[str_predict]] = 1

    for index, tup in enumerate(tuples):
        tup.cid = index
    return tuples