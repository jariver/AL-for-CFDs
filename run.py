import process

path = './data'
file = '/example-dirty.csv'
out_path = './output'
out_file = '/output.txt'

delta = 3

attr_dict, attr_dict_re = process.create_attr_dict(path, file)
tuples = process.create_tuples(path, file, attr_dict)
process.write_data(out_path, out_file, tuples)
dict_freq = process.compute_freq(tuples)
dict_predict, dict_predict_re = process.compute_predict(dict_freq, delta)
tuples = process.compute_feat_vec(tuples, dict_predict, dict_predict_re)

# input_str = input("Which tuple does violate the CFDs you want to express? (Please input the cid of tuples, e.g. 2,3,4,5) >>> ")
input_str = '2,3,4,5'

input_cid = list(map(int, input_str.split(',')))
for cid in input_cid:
    tuples[cid].label = 1

for tup in tuples:
    print(tup.cid, tup.value_dict, tup.feature_vec, tup.label)