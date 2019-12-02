import process
import time
import trainingset
import train

path = './data'
file = '/example-dirty.csv'
out_path = './output'
out_file = '/output.txt'

delta = 2
k = 8

time_start = time.time()
time_end = time.time()
print('begin:', round(time_end - time_start, 2), 's')

attr_dict, attr_dict_re = process.create_attr_dict(path, file)
time_end = time.time()
print('create_attr_dict:', round(time_end - time_start, 2), 's')

tuples = process.create_tuples(path, file, attr_dict)
time_end = time.time()
print('create_tuples:', round(time_end - time_start, 2), 's')

dict_freq = process.compute_freq(tuples)
time_end = time.time()
print('compute_freq:', round(time_end - time_start, 2), 's')

dict_predicate, dict_predicate_re = process.compute_predicate(dict_freq, delta)
time_end = time.time()
print('dict_predicate_re:', round(time_end - time_start, 2), 's')

process.compute_feature_vector(tuples, dict_predicate, dict_predicate_re)
time_end = time.time()
print('compute_feature_vector:', round(time_end - time_start, 2), 's')

process.compute_confidence(tuples)
time_end = time.time()
print('compute_confidence:', round(time_end - time_start, 2), 's')

process.tup_sort(tuples)
time_end = time.time()
print('tup_sort:', round(time_end - time_start, 2), 's')

dataset_list = process.select_tuple(tuples, k)
time_end = time.time()
print('select_tuple:', round(time_end - time_start, 2), 's')

all_predicate_set = process.generate_all_predicate(dict_predicate_re)
time_end = time.time()
print('generate_all_predicate:', round(time_end - time_start, 2), 's')

predicate_list, predicate_list_minus, predicate_list_0 = process.generate_tup_predicate(dataset_list, dict_predicate, all_predicate_set)
time_end = time.time()
print('predicate_list_0:', round(time_end - time_start, 2), 's')

R_CFDs = process.generate_R_CFDs(predicate_list, predicate_list_minus, predicate_list_0)
time_end = time.time()
print('generate_R_CFDs:', round(time_end - time_start, 2), 's')

process.remove_from_R_CFDs(R_CFDs, predicate_list_0)
time_end = time.time()
print('remove_from_R_CFDs:', round(time_end - time_start, 2), 's')

process.clean_from_R_CFDs(R_CFDs, tuples, delta)
time_end = time.time()
print('clean_from_R_CFDs:', round(time_end - time_start, 2), 's')

process.output(R_CFDs, out_path, out_file)
time_end = time.time()
print('output:', round(time_end - time_start, 2), 's')
# training_set, labels = trainingset.generate_training_set(tuples)
# print(training_set, labels)
# train.train(training_set, labels)
