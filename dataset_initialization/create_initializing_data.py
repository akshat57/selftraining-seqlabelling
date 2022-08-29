import pickle
import json
from useful_functions import save_data
import os

def read_ud_dataset(dataset = 'tb', location = '../Datasets/POSTagging/Tweebank/', split = 'train'):
    if dataset == 'tb':
        filename = location + 'en-ud-tweet-' + split + '.conllu'
    elif dataset == 'gum':
        filename = location + 'en_gum-ud-' + split + '.conllu'

    data = []
    tokens = []
    labels = []
    with open(filename, 'r') as file:
        for i, line in enumerate(file):
            line = line[:-1]

            if len(line) and line[0] != '#':
                line = line.split('\t')
                index = line[0]
                if index.find('-') == -1:
                    tokens.append(line[1])
                    labels.append(line[3])

            elif len(line) == 0:
                data.append((tokens, labels))
                tokens = []
                labels = []

    return data


def read_tb_gum():
    tb_location = 'Tweebank/'
    train_tb = read_ud_dataset(dataset = 'tb', location = tb_location, split = 'train')
    dev_tb = read_ud_dataset(dataset = 'tb', location = tb_location, split = 'dev')
    test_tb = read_ud_dataset(dataset = 'tb', location = tb_location, split = 'test')

    gum_location = 'GUM/'
    train_gum = read_ud_dataset(dataset = 'gum', location = gum_location, split = 'train')
    dev_gum = read_ud_dataset(dataset = 'gum', location = gum_location, split = 'dev')
    test_gum = read_ud_dataset(dataset = 'gum', location = gum_location, split = 'test')

    return train_tb, dev_tb, test_tb, train_gum, dev_gum, test_gum

def main():
    save_folder = 'initialization_data/'
    os.makedirs(save_folder, exist_ok = True)

    train_tb, dev_tb, test_tb, train_gum, dev_gum, test_gum = read_tb_gum()

    #save GUM (source) dataset as it is. Add a flag for the tweebank (target) dataset
    save_data(save_folder + 'source_train.pkl', train_gum)
    save_data(save_folder + 'source_dev.pkl', dev_gum)
    save_data(save_folder + 'source_test.pkl', test_gum)

    train_tb = [(sent, labels, [False] * len(labels) ) for (sent, labels) in  train_tb]
    dev_tb = [(sent, labels, [False] * len(labels) ) for (sent, labels) in dev_tb]
    test_tb = [(sent, labels, [False] * len(labels) ) for (sent, labels) in test_tb]

    save_data(save_folder + 'target_train.pkl', train_tb)
    save_data(save_folder + 'target_dev.pkl', dev_tb)
    save_data(save_folder + 'target_test.pkl', test_tb)
    

if __name__ == '__main__':
    main()
