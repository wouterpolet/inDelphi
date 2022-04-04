import pickle as pkl

def read_pickle_file(filename):
    pkl_load = pkl.load(open(filename, 'rb'))
    counts = pkl_load['counts']
    del_features = pkl_load['del_features']

    # Ignoring the Fraction Column
    counts = counts.drop("fraction", axis=1)
    return counts, del_features

if __name__ == '__main__':
    counts, del_features = read_pickle_file('../in/dataset.pkl')
