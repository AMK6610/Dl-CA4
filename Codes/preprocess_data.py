import numpy as np
import pickle


data_directory = '../data/'
train_file = 'Train_Arabic_Digit.txt'
test_file = 'Test_Arabic_Digit.txt'
saved_file = 'blocked_data.pkl'

bsize_train = 660
bsize_test = 220
mfcc_len = 13
N = 2
before_after = 9

def derivative(vector):
    dt = np.zeros(mfcc_len)
    s = 2 * np.sum(np.array([k + 1 for k in range(N)]) ** 2)
    for t in range(mfcc_len):
        d = 0
        for j in [k + 1 for k in range(N)]:
            idx1 = t - j if t - j >= 0 else 0
            idx2 = t + j if t + j < mfcc_len else mfcc_len - 1
            d += j * (vector[idx2] - vector[idx1])
        dt[t] = d / s
    return dt

def prepare_data():
    partition = {
        'train_data': [],
        'train_label': [],
        'test_data': [],
        'test_label': [],
        'train_parts': [],
        'test_parts': []
    }
    
    for data_file in [('train', train_file), ('test', test_file)]:
        i = 0
        index = 0
        prev_label = 0
        block = []
        # label_block = []
        with open(data_directory + data_file[1]) as file:
            line = file.readline()
            for line in file:
                if not line.split():
                    i += 1
                    partition[data_file[0] + '_data'].append(block)
                    # partition[data_file[0] + '_label'].append(label_block)
                    block = []
                    # label_block = []
                else:
                    datum = np.array(line.split(), dtype=float)
                    datum = np.concatenate((datum, derivative(datum), derivative(derivative(datum))))
                    # print(datum, datum.shape)
                    label = i // bsize_train if data_file[0] == 'train' else i // bsize_test
                    if label != prev_label:
                      partition[data_file[0] + '_parts'].append(index)
                    block.append(datum)
                    partition[data_file[0] + '_label'].append(label)
                    # label_block.append(label)
                    prev_label = label
                    index += 1
                # print(datum)
                # print(np.array(partition['train_data']).shape)
                # print(line)
        partition[data_file[0] + '_data'].append(block)
        # partition[data_file[0] + '_label'].append(label_block)
        block = []
        # label_block = []
    partition['train_parts'].append(len(partition['train_label']))
    partition['test_parts'].append(len(partition['test_label']))
    pickle.dump(partition, open(data_directory + saved_file, 'wb'))
    return partition

def prepare_dnn_data():
    partition = pickle.load(open(data_directory + saved_file, 'rb'))
    partition['train_dnn_data'] = []
    partition['test_dnn_data'] = []
    partition['train_dnn_label'] = []
    partition['test_dnn_label'] = []
    for data_type in ['train', 'test']:
        for i, block in enumerate(partition[data_type + '_data']):
            print(len(block))
            start = 0
            while start + before_after < len(block):
                partition[data_type + '_dnn_data'].append(np.array(block[start: start + before_after]).ravel())
                label = i // bsize_train if data_type == 'train' else i // bsize_test
                partition[data_type + '_dnn_label'].append(label)
                start += 1
    print(partition['train_dnn_data'][0].shape)
    pickle.dump(partition, open(data_directory + saved_file, 'wb'))




    


def main():
    # prepare_data()
    prepare_dnn_data()

    # X1 = [[0.5], [1.0], [-1.0], [0.42], [0.24]]
    # X2 = [[2.4], [4.2], [0.5], [-0.24]]
    # X = np.concatenate([X1, X2])
    # lengths = [len(X1), len(X2)]
    # print(np.array(X1).shape, np.array(X2).shape, lengths, X.shape, X)

    partition = pickle.load(open(data_directory + saved_file, 'rb'))
    print(np.array(partition['train_data']).shape, np.array(partition['train_label']).shape, np.array(partition['test_data']).shape, np.array(partition['test_label']).shape)
    # print(partition['test_parts'], partition['train_parts'])
    # print(partition['train_label'][partition['train_parts'][8]: partition['train_parts'][9]])


if __name__ == '__main__':

    main()