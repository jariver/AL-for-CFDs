import process
import trainingset
import train

path = './data'
file = '/example-dirty.csv'
out_path = './output'
out_file = '/output.txt'

delta = 3
k = 6

attr_dict, attr_dict_re = process.create_attr_dict(path, file)
tuples = process.create_tuples(path, file, attr_dict)
dict_freq = process.compute_freq(tuples)
dict_predicate, dict_predicate_re = process.compute_predicate(dict_freq, delta)
process.compute_feature_vector(tuples, dict_predicate, dict_predicate_re)
process.compute_confidence(tuples)
process.tup_sort(tuples)
dataset_list = process.select_tuple(tuples, k)
all_predicate_set = process.generate_all_predicate(dict_predicate_re)
predicate_list, predicate_list_minus, predicate_list_0 = process.generate_tup_predicate(dataset_list, dict_predicate, all_predicate_set)
R_CFDs = process.generate_R_CFDs(predicate_list, predicate_list_minus)

process.remove_from_R_CFDs(R_CFDs, predicate_list_0)
process.clean_from_R_CFDs(R_CFDs)
process.output(R_CFDs, out_path, out_file)
# training_set, labels = trainingset.generate_training_set(tuples)
# print(training_set, labels)
# train.train(training_set, labels)
