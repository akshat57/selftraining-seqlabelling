from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
import numpy as np
import torch


class dataset(Dataset):
  def __init__(self, all_data, tokenizer, labels_to_ids, max_len):
        self.len = len(all_data)
        self.data = all_data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels_to_ids = labels_to_ids

  def __getitem__(self, index):
        # step 1: get the sentence and word labels 
        sentence = self.data[index][0]
        joined_sentnece = ' '.join(sentence)
        word_labels = self.data[index][1]

        # step 2: use tokenizer to encode sentence (includes padding/truncation up to max length)
        # BertTokenizerFast provides a handy "return_offsets_mapping" functionality for individual tokens
        encoding = self.tokenizer(sentence,
                             is_split_into_words=True, #no is_pretokenlized(Modification), we already have a splitted sentence
                             return_offsets_mapping=True, 
                             padding='max_length', 
                             truncation=True, 
                             max_length=self.max_len)

        # step 3: create token labels only for first word pieces of each tokenized word
        #labels = [self.labels_to_ids[label] for label in word_labels] 
        labels = [self.labels_to_ids[label] for label in word_labels if label in list(self.labels_to_ids.keys())] 
        # code based on https://huggingface.co/transformers/custom_datasets.html#tok-ner
        # create an empty array of -100 of length max_length

        encoded_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100
        
        # set only labels whose first offset position is 0 and the second is not 0
        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
          if mapping[0] == 0 and mapping[1] != 0:
            # overwrite label
            if i >= len(labels):
              break
            encoded_labels[idx] = labels[i]
            i += 1
            if i == len(labels):
              break

        # step 4: turn everything into PyTorch tensors
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.as_tensor(encoded_labels)

        '''print(sentence)
        print(item['input_ids'])
        print('0:', item['offset_mapping'][:,0])
        print('1:', item['offset_mapping'][:,1])
        print()'''
        
        #return original data alongwith tensor data
        original_data = {}
        if len(self.data[index]) == 3:
          original_data['sentence'] = sentence
          original_data['labels'] = word_labels
          original_data['used_flag'] = self.data[index][2]

        
        return item, original_data


  def __len__(self):
        return self.len


def collate_function(tuple_lst):

    batch_size = len(tuple_lst)
    item_list = [x[0] for x in tuple_lst]
    original_data_list = [x[1] for x in tuple_lst]
    max_len = item_list[0]['input_ids'].shape[0]

    #convert list of dictionary to dictionary with values as list
    input_ids = torch.zeros([batch_size, max_len], dtype=torch.int64)
    attention_mask = torch.zeros([batch_size, max_len], dtype=torch.int64)
    labels = torch.zeros([batch_size, max_len], dtype=torch.int64)

    for i, element in enumerate(item_list):
      input_ids[i] = element['input_ids']
      attention_mask[i] = element['attention_mask']
      labels[i] = element['labels']

    item = {}
    item['input_ids'] = input_ids
    item['attention_mask'] = attention_mask
    item['labels'] = labels

    original_data = {'sentence': [], 'labels': [], 'used_flag': []}
    if len(original_data_list[0].keys() ):
  
      for element in original_data_list:
        original_data['sentence'].append(element['sentence'])
        original_data['labels'].append(element['labels'])
        original_data['used_flag'].append(element['used_flag'])

    return item, original_data


def initialize_data(tokenizer, initialization_input, input_data):
  max_len, train_batch_size, dev_batch_size, test_batch_size = initialization_input
  train_data, dev_data, test_data, labels_to_ids_train, labels_to_ids_dev, labels_to_ids_test  = input_data

  training_set = dataset(train_data, tokenizer, labels_to_ids_train, max_len)
  validation_set = dataset(dev_data, tokenizer, labels_to_ids_dev, max_len)
  testing_set = dataset(test_data, tokenizer, labels_to_ids_test, max_len)

  train_params = {'batch_size': train_batch_size,
              'shuffle': True,
              'num_workers': 4
              }

  dev_params = {'batch_size': dev_batch_size,
              'shuffle': False,
              'num_workers': 4
              }

  test_params = {'batch_size': test_batch_size,
              'shuffle': False,
              'num_workers': 4
              }

  train_loader = DataLoader(training_set, collate_fn=collate_function, **train_params)
  dev_loader = DataLoader(validation_set, collate_fn=collate_function, **dev_params)
  test_loader = DataLoader(testing_set, collate_fn=collate_function, **test_params)

  return train_loader, dev_loader, test_loader




if __name__ == '__main__':
    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 4
    VALID_BATCH_SIZE = 2
    EPOCHS = 1
    LEARNING_RATE = 1e-05
    MAX_GRAD_NORM = 10
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased') #Bert using wordpiece *** may improve further

    train_file = 'twi/train.txt'
    test_file = 'twi/test.txt'
    dev_file = 'twi/dev.txt'

    dev_data = read_pos_standard(dev_file)
    train_data = read_pos_standard(train_file)
    test_data = read_pos_standard(test_file)

    ##Create labels to index mapping
    labels_to_ids = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-LOC': 3, 'I-LOC': 4, 'B-ORG': 5, 'I-ORG': 6, 'B-DATE':7, 'I-DATE': 8}
    ids_to_labels = dict((v,k) for k,v in labels_to_ids.items())
    ####end mapping

    #training_set = dataset(train_data, tokenizer, labels_to_ids, MAX_LEN)

    '''df_train = get_sentence_labels(df_train_raw)
    df_valid = get_sentence_labels(df_valid_raw)
    df_test = get_sentence_labels(df_test_raw)'''


