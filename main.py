import os
import time

#Create a new repository called iteration 0. Inside that create a repo called logs.
#It should contain 'processed_data_0.pkl': this is the entire dataset initially. This dataset could be binary or trinary class.


for i in range(1, 20):
    start = time.time()
    iter = str(i)

    print('='*20)
    print()
    print('Working on iteration ' + iter)
    
    #FINE TUNE MODEL (FINE TUNE BLOCK)
    command = 'python3 pos_training_code.py --iter ' + iter
    print('\nFine Tuning:', command)
    os.system(command)
    #os.system(command + ' > iteration' + iter + '/logs/training_')

    #MAKING PREDICTIONS FROM FINE TUNED MODEL (PREDICTION BLOCK)
    command = 'python3 pseudo_labelling.py --iter ' + iter 
    print('\nLoaded prediction:', command)
    os.system(command)
    #os.system( command + ' > iteration' + iter + '/logs/prediction_load_log')
    
    
    print('Time for 1 epoch:', time.time() - start)
    print()



