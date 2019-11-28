import csv
import operator
from Tuple import *
from DataSet import *


def create_attr_dict(path, file):
    r"""Create two dictionaries using to mark the index of Attributes.

    Args:
        path (str): Dictionary of the csv file.
        file (str): The name of the csv file.

    Return:
        attr_dict (dict): {index: attribute, ...}, e.g.{6: 'ZIP', ...}.
        attr_dict_re (dict): {attribute: index, ...}, e.g.{'ZIP': 6, ...}.

    """
    with open(path+file, encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        attr_name = next(reader)
        attr_dict = {i: attr for i, attr in enumerate(attr_name)}
        attr_dict_re = {attr: i for i, attr in enumerate(attr_name)}
    return attr_dict, attr_dict_re


def create_tuples(path, file, attr_dict):
    r"""Create a list of Tuple Class.

    Args:
        path (str): Dictionary of the csv file.
        file (str): The name of the csv file.
        attr_dict (dict): {index: attribute, ...}, e.g.{6: 'ZIP', ...}.

    Return:
        tuples (list): A list of all instances of Tuple Class, which contains all tuples in whole csv file.
    """
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
    r"""Compute the frequency of all values.

    Args:
        tuples: A list of all instances of Tuple Class, which contains all tuples in whole csv file.

    Return:
        dict_freq: {'ZIP': {'07974': 4, '07975': 5, ...}, 'CT': {'MH': 2, ...}, ...}
    """
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


def compute_predicate(dict_freq, delta):
    r"""
    delta = 2
    predicate can be selected if its frequent >= 2
    """
    dict_predicate = dict()
    dict_predicate_re = dict()

    index = 0
    for attr, dict_freq_dict in dict_freq.items():
        for key, value in dict_freq_dict.items():
            if value >= delta:
                str_predicate = str(attr) + '=' + str(key)
                dict_predicate[index] = str_predicate
                dict_predicate_re[str_predicate] = index
                index += 1
    return dict_predicate, dict_predicate_re


def compute_feature_vector(tuples, dict_predicate, dict_predicate_re):
    r"""Compute the feature vector of every tuple and generate the cid.
    """
    for tup in tuples:
        for i in dict_predicate.keys():
            tup.feature_vec.append(0)

        for attr, value in tup.value_dict.items():
            str_predicate = str(attr) + '=' + str(value)
            if str_predicate in dict_predicate_re.keys():
                tup.feature_vec[dict_predicate_re[str_predicate]] = 1

    for index, tup in enumerate(tuples):
        tup.cid = index


def compute_confidence(tuples):
    r"""Compute the confidence of a tuple. confidence = (#predicates in tuple) / (#all predicates)

    Args:
        tuples (list): A list of all instances of Tuple Class, which contains all tuples in whole csv file.
    """
    for tup in tuples:
        tup_predicate_count = tup.feature_vec.count(1)
        all_predicate_count = len(tup.feature_vec)
        tup.confidence = tup_predicate_count / all_predicate_count


def mark_label(true_str, false_str, tuples):
    r"""

    Args:
        true_str (str): A cid string user input. These tuples violate the CFDs user wants to express, e.g.'0,1,7'
        false_str (str): A cid string user input. These tuples don't violate the CFDs user wants to express, e.g.'3'
        tuples (list): A list of all instances of Tuple Class, which contains all tuples in whole csv file.
    """
    true_cid = list(map(int, true_str.split(',')))
    for cid in true_cid:
        for tup in tuples:
            if tup.cid == cid:
                tup.label = 1
                break

    false_cid = list(map(int, false_str.split('.')))
    for cid in false_cid:
        for tup in tuples:
            if tup.cid == cid:
                tup.label = 0
                break


def tup_sort(tuples):
    r"""Sort in descending order of confidence in Tuple instance.

    Args:
        tuples:
    """
    tuples.sort(key=operator.attrgetter('confidence'), reverse=True)


def select_tuple(tuples, k):
    r"""

    Args:
        tuples:
        k:
    """
    for i in range(k):
        print(tuples[i].cid, tuples[i].value_dict)

    # true_str = input("Which tuples violate the CFDs you want to express? "
    #                    "(Please input the cid of tuples, e.g. 2,3,4,5) >>> ")
    # false_str = input("Which tuple don't violate the CFDs you want to express? "
    #                    "(Please input the cid of tuples, e.g. 2,3,4,5) >>> ")
    true_str = '0,1'
    false_str = '3'

    mark_label(true_str, false_str, tuples)

    dataset_list = list()
    for i in range(k):
        dataset = DataSet()
        dataset.cid = tuples[i].cid
        dataset.feature_vec = tuples[i].feature_vec
        dataset.label = tuples[i].label
        dataset_list.append(dataset)
    return dataset_list

