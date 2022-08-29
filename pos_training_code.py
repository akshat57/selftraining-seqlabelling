import pandas as pd
import numpy as np
import torch
from torch import cuda
from torch.utils.data import Dataset, DataLoader
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.metrics import accuracy_score, classification_report
from load_data import initialize_data
from labels_to_ids import tweebank_labels_to_ids
import time
import os
from useful_functions import load_data, save_data
import argparse


def train(epoch, training_loader, model, optimizer, device, grad_step = 1, max_grad_norm = 10):
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_preds, tr_labels = [], []
    # put model in training mode
    model.train()
    optimizer.zero_grad()
    
    for idx, (batch, original_data) in enumerate(training_loader):
        ids = batch['input_ids'].to(device, dtype = torch.long)
        mask = batch['attention_mask'].to(device, dtype = torch.long)
        labels = batch['labels'].to(device, dtype = torch.long)

        #loss, tr_logits = model(input_ids=ids, attention_mask=mask, labels=labels)
        output = model(input_ids=ids, attention_mask=mask, labels=labels)
        tr_loss += output[0]

        nb_tr_steps += 1
        nb_tr_examples += labels.size(0)
           
        # compute training accuracy
        flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
        active_logits = output[1].view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
        
        # only compute accuracy at active labels
        active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
        #active_labels = torch.where(active_accuracy, labels.view(-1), torch.tensor(-100).type_as(labels))
        
        labels = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)
        
        tr_labels.extend(labels)
        tr_preds.extend(predictions)

        tmp_tr_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
        tr_accuracy += tmp_tr_accuracy
    
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=max_grad_norm
        )
        
        # backward pass
        output['loss'].backward()
        if (idx + 1) % grad_step == 0:
            optimizer.step()
            optimizer.zero_grad()

    epoch_loss = tr_loss / nb_tr_steps
    tr_accuracy = tr_accuracy / nb_tr_steps
    #print(f"Training loss epoch: {epoch_loss}")
    #print(f"Training accuracy epoch: {tr_accuracy}")

    return model


def testing(model, testing_loader, labels_to_ids, device):
    # put model in evaluation mode
    model.eval()
    
    eval_loss, eval_accuracy = 0, 0
    nb_eval_examples, nb_eval_steps = 0, 0
    eval_preds, eval_labels = [], []
    
    ids_to_labels = dict((v,k) for k,v in labels_to_ids.items())

    with torch.no_grad():
        for idx, (batch, original_data) in enumerate(testing_loader):
            
            ids = batch['input_ids'].to(device, dtype = torch.long)
            mask = batch['attention_mask'].to(device, dtype = torch.long)
            labels = batch['labels'].to(device, dtype = torch.long)
            
            #loss, eval_logits = model(input_ids=ids, attention_mask=mask, labels=labels)
            output = model(input_ids=ids, attention_mask=mask, labels=labels)

            eval_loss += output['loss'].item()

            nb_eval_steps += 1
            nb_eval_examples += labels.size(0)
        
            if idx % 100==0:
                loss_step = eval_loss/nb_eval_steps
                print(f"Validation loss per 100 evaluation steps: {loss_step}")
              
            # compute evaluation accuracy
            flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
            active_logits = output[1].view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
            
            # only compute accuracy at active labels
            active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
        
            labels = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)
            
            eval_labels.extend(labels)
            eval_preds.extend(predictions)
            
            tmp_eval_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
            eval_accuracy += tmp_eval_accuracy

    labels = [ids_to_labels[id.item()] for id in eval_labels]
    predictions = [ids_to_labels[id.item()] for id in eval_preds]
    
    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_steps
    #print(f"Validation Loss: {eval_loss}")
    #print(f"Validation Accuracy: {eval_accuracy}")

    return labels, predictions, eval_accuracy


def load_train_data(data_location, data_type = 'source'):
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
    n_epochs = 8
    model_name = 'bert-base-uncased'
    max_len = 256
    train_batch_size = 8
    dev_batch_size = 8
    test_batch_size = 8
    grad_step = 4
    learning_rate = 1e-05
    initialization_input = (max_len, train_batch_size, dev_batch_size, test_batch_size)

    #model saving parameters
    model_save_flag = True
    model_save_location = 'iteration' + args.iter + '/saved_models/' + model_name.replace('/', '-')
    os.makedirs(model_save_location, exist_ok = True)
    logfile_location = 'iteration' + args.iter + '/logs/logs_training.txt'
    f = open(logfile_location, 'w')
    f.close()

    #Load data
    dataset_location = 'iteration' + args.iter + '/data/'
    source_train, source_dev, source_test, train_labels, dev_labels, test_labels = load_train_data(dataset_location, 'source')
    print(len(source_train))
    source_input_data = (source_train, source_dev, source_test, train_labels, dev_labels, test_labels)
    target_train, target_dev, target_test, train_labels, dev_labels, test_labels = load_train_data(dataset_location, 'target')
    target_input_data = (target_train, target_dev, target_test, train_labels, dev_labels, test_labels)

    #Define tokenizer, model and optimizer
    device = 'cuda' if cuda.is_available() else 'cpu' #save the processing time
    tokenizer =  AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(train_labels))
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    model.to(device)

    #Get dataloaders
    train_loader, dev_loader, test_loader = initialize_data(tokenizer, initialization_input, source_input_data)
    target_train_loader, target_dev_loader, target_test_loader = initialize_data(tokenizer, initialization_input, target_input_data)

            
    best_dev_acc = 0
    best_test_acc = 0
    best_epoch = -1
    for epoch in range(n_epochs):
        start = time.time()
        print(f"Training epoch: {epoch + 1}")

        #train model
        model = train(epoch, train_loader, model, optimizer, device, grad_step)
        
        #testing and logging
        labels_dev, predictions_dev, dev_accuracy = testing(model, dev_loader, dev_labels, device)
        print('DEV ACC:', dev_accuracy)
        
        labels_test, predictions_test, test_accuracy = testing(model, test_loader, test_labels, device)
        print('TEST ACC:', test_accuracy)

        labels_test_tb, predictions_test_tb, test_accuracy_tb = testing(model, target_test_loader, test_labels, device)
        print('TARGET TEST ACC:', test_accuracy_tb)
        
        #saving model
        if dev_accuracy > best_dev_acc:
            best_dev_acc = dev_accuracy
            best_test_acc = test_accuracy
            best_epoch = epoch
            
            if model_save_flag:
                os.makedirs(model_save_location, exist_ok=True)
                tokenizer.save_pretrained(model_save_location)
                model.save_pretrained(model_save_location)


        #logging
        f = open(logfile_location, 'a')
        f.write('EPOCH: ' + str(epoch) + '\n\n')
        f.write('SOURCE TEST:' + '\n\n')
        f.write(classification_report(labels_test, predictions_test, digits = 5))
        f.write('\n\nTARGET TEST:' + '\n\n')
        f.write(classification_report(labels_test_tb, predictions_test_tb, digits = 5))
        f.write('\nDEV ACC : ' + str(round(dev_accuracy, 5)) + '\n')
        f.write('TEST ACC : ' + str(round(test_accuracy, 5)) + '\n')
        f.write('TARGET TEST ACC : ' + str(round(test_accuracy_tb, 5)) + '\n')
        f.write('BEST EPOCH : ' + str(best_epoch) + '\n')
        f.write('BEST ACCURACY --> ' +  'DEV:' +  str(round(best_dev_acc, 5)) + ', TEST:' + str(round(best_test_acc, 5)) + '\n')
        f.write('-'*80 + '\n')
        f.close()

        now = time.time()
        print('BEST ACCURACY --> ', 'DEV:', round(best_dev_acc, 5), 'TEST:',  round(best_test_acc, 5))
        print('TIME PER EPOCH:', (now-start)/60 )
        print()