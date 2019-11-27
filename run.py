import process
# import trainingset
# import train

path = './data'
file = '/example-dirty.csv'
out_path = './output'
out_file = '/output.txt'

delta = 3

attr_dict, attr_dict_re = process.create_attr_dict(path, file)
tuples = process.create_tuples(path, file, attr_dict)
process.write_data(out_path, out_file, tuples)
dict_freq = process.compute_freq(tuples)
dict_predicate, dict_predicate_re = process.compute_predicate(dict_freq, delta)
tuples = process.compute_feature_vector(tuples, dict_predicate, dict_predicate_re)
tuples = process.compute_confidence(tuples)

# input_str = input("Which tuple does violate the CFDs you want to express? "
#                   "(Please input the cid of tuples, e.g. 2,3,4,5) >>> ")
true_str = '0,1,7'
false_str = '3'

tuples = process.mark_label(true_str, false_str, tuples)

# training_set, labels = trainingset.generate_training_set(tuples)
# train.train(training_set, labels)

for tup in tuples:
    print(tup.cid, tup.value_dict, tup.feature_vec, tup.label, tup.confidence)
