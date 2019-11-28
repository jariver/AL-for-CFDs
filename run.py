import process
import trainingset
import train

path = './data'
file = '/example-dirty.csv'
out_path = './output'
out_file = '/output.txt'

delta = 3
k = 3

attr_dict, attr_dict_re = process.create_attr_dict(path, file)
tuples = process.create_tuples(path, file, attr_dict)
process.write_data(out_path, out_file, tuples)
dict_freq = process.compute_freq(tuples)
dict_predicate, dict_predicate_re = process.compute_predicate(dict_freq, delta)
process.compute_feature_vector(tuples, dict_predicate, dict_predicate_re)
process.compute_confidence(tuples)
process.tup_sort(tuples)
dataset_list = process.select_tuple(tuples, k)

training_set, labels = trainingset.generate_training_set(tuples)
print(training_set, labels)
train.train(training_set, labels)
