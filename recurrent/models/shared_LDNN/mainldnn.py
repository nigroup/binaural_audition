import sys
sys.path.insert(0, '/home/changbinli/script/rnn/')
import pandas as pd
from architecture.standard_ldnn import HyperParameters
import tensorflow as tf

if __name__ == "__main__":
    validation_fold = 3
    # read params
    df = pd.read_csv('/home/changbinli/script/rnn/randomsearch/hyper_combinations_7.13.csv')
    # folder_name = 'sub_0'
    # with tf.Graph().as_default():
    #     hyperparameters = HyperParameters(VAL_FOLD=validation_fold, FOLD_NAME=folder_name)
    #     hyperparameters.main()

    for index in range(0,20):
        print('Training combination#',index)
        dic = {}
        dic['LEARNING_RATE'] = df.loc[index]['LEARNING_RATE']
        dic['LAMBDA_L2'] = df.loc[index]['LAMBDA_L2']
        dic['OUTPUT_KEEP_PROB'] = df.loc[index]['OUTPUT_KEEP_PROB']
        dic['NUM_LSTM'] = int(df.loc[index]['NUM_LSTM'])
        dic['NUM_HIDDEN'] = int(df.loc[index]['NUM_HIDDEN'])
        dic['NUM_MLP'] = int(df.loc[index]['NUM_MLP'])
        dic['NUM_NEURON'] = int(df.loc[index]['NUM_NEURON'])
        dic['BATCH_SIZE'] = int(df.loc[index]['BATCH_SIZE'])
        dic['TIMELENGTH'] = int(df.loc[index]['BATCH_LENGTH'])

        folder_name = 'standard_' + str(index)
        with tf.Graph().as_default():
            hyperparameters = HyperParameters(VAL_FOLD=validation_fold, FOLD_NAME=folder_name)
            for key, value in dic.items():
                setattr(hyperparameters, key, value)
            hyperparameters.update_attribute()
            hyperparameters.main()

        print("Combination #",index,"finished!")