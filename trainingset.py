def generate_training_set(tuples):
    training_set = list()
    labels = list()

    for tup in tuples:
        if tup.label is not None:
            training_set.append(tup.feature_vec)
            labels.append(tup.label)
    return training_set, labels

def update_train_set():
    pass
