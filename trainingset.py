def generate_training_set(dataset_list):
    training_set = list()
    labels = list()

    for dataset in dataset_list:
        if dataset.label is not None:
            training_set.append(dataset.feature_vec)
            labels.append([dataset.label])
    return training_set, labels

def update_train_set():
    pass
