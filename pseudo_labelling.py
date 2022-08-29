import pandas as pd
import numpy as np
import torch
from torch import cuda
from torch.utils.data import Dataset, DataLoader
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.metrics import accuracy_score, classification_report
from load_data import initialize_data
from labels_to_ids import tweebank_labels_to_ids, tweebank_ids_to_labels
import time
import os
from useful_functions import load_data, save_data
import argparse

def testing(model, testing_loader, labels_to_ids, device, logfile_location):
    self_training_threshold = 0.97

    # put model in evaluation mode
    model.eval()
    
    eval_loss, eval_accuracy = 0, 0
    nb_eval_examples, nb_eval_steps = 0, 0
    eval_preds, eval_labels = [], []
    
    ids_to_labels = dict((v,k) for k,v in labels_to_ids.items())

    softmax = torch.nn.Softmax(dim=-1)

    all_averages = torch.empty(0).to(device)
    all_sentences = []
    all_labels = []
    all_flags = []#doing nothing with flags here

    threshold = 0.97

    logfile = open(logfile_location, 'w')

    with torch.no_grad():
        for idx, (batch, original_data) in enumerate(testing_loader):
            
            ids = batch['input_ids'].to(device, dtype = torch.long)
            mask = batch['attention_mask'].to(device, dtype = torch.long)
            labels = batch['labels'].to(device, dtype = torch.long)

            all_sentences.extend(original_data['sentence'])
            #all_labels.extend(original_data['labels'])
            all_flags.extend(original_data['used_flag'])
            
            #loss, eval_logits = model(input_ids=ids, attention_mask=mask, labels=labels)
            output = model(input_ids=ids, attention_mask=mask, labels=labels)
              

            # compute evaluation accuracy
            flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
            active_logits = output[1]#.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=2) # shape (batch_size * seq_len,)


            #get pseudo labels
            for i in range(flattened_predictions.shape[0]):
                pseudo_labels_ids = flattened_predictions[i,:][labels[i,:] != -100]
                pseudo_labels = [tweebank_ids_to_labels[token.item()] for token in pseudo_labels_ids]
                all_labels.append(pseudo_labels)

            prediction_values, _ = torch.max(softmax(active_logits), axis=2) # shape (batch_size * seq_len,)

            prediction_values = prediction_values * mask
            average_scores = torch.sum(prediction_values, dim = 1) / torch.count_nonzero(prediction_values, dim = 1)
            all_averages = torch.cat((all_averages, average_scores), dim = 0)
            
    print(torch.mean(all_averages).item(), torch.std(all_averages).item(), torch.max(all_averages).item())
    logfile.write(str(torch.mean(all_averages).item()) + ',' + str(torch.std(all_averages).item()) + ',' + str(torch.max(all_averages).item()) + '\n')


    selected_indices = all_averages >= self_training_threshold
    selected_indices = selected_indices.detach().cpu().tolist()

    unselected_indices  = all_averages < self_training_threshold
    unselected_indices = unselected_indices.detach().cpu().tolist()

    #choose selected data
    selected_sentences = [all_sentences[i] for i, select in enumerate(selected_indices) if select]
    selected_labels = [all_sentences[i] for i, select in enumerate(selected_indices) if select]
    selected_flags = [all_sentences[i] for i, select in enumerate(selected_indices) if select]

    #choose selected data
    unselected_sentences = [all_sentences[i] for i, select in enumerate(selected_indices) if not select]
    unselected_labels = [all_sentences[i] for i, select in enumerate(selected_indices) if not select]
    unselected_flags = [all_sentences[i] for i, select in enumerate(selected_indices) if not select]

    print('SELECTED SENTENCES:'  +  str(len(selected_sentences)) + '\n')
    logfile.write('SELECTED SENTENCES:'  +  str(len(selected_sentences)) + '\n')
    logfile.close()
    return selected_sentences, selected_labels, selected_flags, unselected_sentences, unselected_labels, unselected_flags 

def load_train_data(data_location, data_type = 'target'):
    train_data = load_data(data_location + data_type + '_train.pkl')
    dev_data = load_data(data_location + data_type + '_dev.pkl')
    test_data = load_data(data_location + data_type + '_test.pkl')

    train_labels = tweebank_labels_to_ids
    dev_labels = tweebank_labels_to_ids
    test_labels = tweebank_labels_to_ids

    return train_data, dev_data, test_data, train_labels, dev_labels, test_labels



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enter iteration number')
    parser.add_argument('--iter', type=str, required=True, help='Enter Iteration number')
    args = parser.parse_args()
    print('Entered', args.iter)

    #Initialization training parameters
    n_epochs = 15
    model_name = 'bert-base-uncased'
    max_len = 256
    train_batch_size = 8
    dev_batch_size = 8
    test_batch_size = 8
    grad_step = 4
    learning_rate = 1e-05
    initialization_input = (max_len, train_batch_size, dev_batch_size, test_batch_size)

    #model saving parameters
    model_load_location = 'iteration' + args.iter + '/saved_models/' + model_name.replace('/', '-')
    logfile_location = 'iteration' + args.iter + '/logs/logs_prediction.txt'

    #Load data
    dataset_location = 'iteration' + args.iter + '/data/'
    source_train, source_dev, source_test, train_labels, dev_labels, test_labels = load_train_data(dataset_location, 'source')
    target_train, target_dev, target_test, train_labels, dev_labels, test_labels = load_train_data(dataset_location, 'target')
    input_data = (target_train, target_dev, target_test, train_labels, dev_labels, test_labels)

    #Define tokenizer, model and optimizer
    device = 'cuda' if cuda.is_available() else 'cpu' #save the processing time
    tokenizer = AutoTokenizer.from_pretrained(model_load_location)
    model = AutoModelForTokenClassification.from_pretrained(model_load_location)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    model.to(device)

    #Get dataloaders
    train_loader, dev_loader, test_loader = initialize_data(tokenizer, initialization_input, input_data)
        
    selected_sentences, selected_labels, selected_flags, unselected_sentences, unselected_labels, unselected_flags = testing(model, test_loader, test_labels, device, logfile_location)


    ##Create data for next iteration
    new_iteration = 'iteration' + str(int(args.iter) + 1) + '/'
    save_data_location = new_iteration + 'data/'
    log_location = new_iteration + 'logs/'
    
    os.makedirs(new_iteration, exist_ok = True)
    os.makedirs(save_data_location, exist_ok = True)
    os.makedirs(log_location, exist_ok = True)


    additional_source_data = []
    for (sentence, labels) in zip(selected_sentences, selected_labels):
        additional_source_data.append((sentence, labels))
    new_source_train = source_train + additional_source_data

    new_target_data = []
    for (sentence, labels, flags) in zip(unselected_sentences, unselected_labels, unselected_flags):
        new_target_data.append((sentence, labels, flags))

    
    #Save data for next iteration 
    save_data(save_data_location+ 'source_train.pkl', new_source_train)
    save_data(save_data_location + 'source_dev.pkl', source_dev)
    save_data(save_data_location + 'source_test.pkl', source_test)

    save_data(save_data_location + 'target_train.pkl', new_target_data)
    save_data(save_data_location + 'target_dev.pkl', target_dev)
    save_data(save_data_location + 'target_test.pkl', target_test)
    
    





    