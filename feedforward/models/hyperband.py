#from cnnmodel import *
import socket
from readData import *
from settings import *
from model import *
from hyperparams import *


trainData = DataSet(trainDir,frames=framelength,folds=trainFolds, overlapSampleSize=25, shortload=shortload)
testData = DataSet(testDir,frames=framelength,folds=testFolds, overlapSampleSize=25, shortload=shortload)







def kfold(hyperparams, data,graphModel, g):

    avg_acc=0

    for k in trainFolds:
        with tf.Session(graph=g) as sess:
            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init)

            copytrainFolds = trainFolds[:]
            copytrainFolds.remove(k)
            data.groupFolds(trainFolds=copytrainFolds,valFolds=[k],testFolds=[])

            data.calcweightsOnTrainFolds()
            data.standardize()



            #also calc standardize matrix here

            acc= train(hyperparams,data,sess,graphModel)
            avg_acc= acc + avg_acc
    return avg_acc/k



def validate_mean_over_instances(data, graphModel,sess,  type):  #data is a list of lists of a dict { x y_block}
    if type=="val":
        numberScenes = len(data.valX)
    else:
        numberScenes= len(data.testX)

    acc = 0
    for instance in np.arange(numberScenes):
        data_x, data_y = data.getData(type, instance)
        o_recall = sess.run([graphModel.recall], feed_dict={graphModel.y: data_y, graphModel.x: data_x})
        acc_instance = o_recall[0]


        acc = acc + acc_instance

    average_acc_batch = acc / numberScenes
    return average_acc_batch






def train(hyperparams ,data, sess,graphModel):
    bestacc = 0

    data.calcNumberOfBatches(batchsize=hyperparams["batchsize"])

    for epoch in range(hyperparams["epochs_per_k_fold_cross_validation"]):

        # training on all apart from k
        for i in range(data.batches):
            train_x, train_y = data.get_next_train_batch()
            sess.run([graphModel.optimiser], feed_dict={graphModel.x: train_x, graphModel.y: train_y, graphModel.cross_entropy_class_weights : data.cross_entropy_class_weights})


            #validation #
            if i%5==0 and len(data.valFolds)>0:
                acc = validate_mean_over_instances(data, graphModel, sess, "val")
                print(acc)
                if acc > bestacc:
                    bestacc = acc

    return bestacc


hyperparamClass = Hyperparams()
hp_acc = []
for i in np.arange(2):
    hyperparams = hyperparamClass.getworkingHyperparams()

    print("train hyperparameter configuration" + str(i))
    with tf.Graph().as_default() as g:
        graphModel = GraphModel(hyperparams)
        single_acc = kfold(hyperparams, trainData, graphModel, g)
        hp_acc.append(kfold(hyperparams, trainData, graphModel, g))


hp_acc = np.array(hp_acc)
best_hyperparams = np.argmax(hp_acc)




best_hyperparams = 1

#final training and testing
with tf.Graph().as_default() as gFinalTrain:
    graphModel = GraphModel(hyperparameterlist[best_hyperparams])
    with tf.Session(graph=gFinalTrain) as sess:
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)

        trainData.groupFolds(trainFolds=trainFolds, valFolds=[], testFolds=[])
        trainData.calcweightsOnTrainFolds()
        trainData.standardize()
        print("I start training")
        train(hyperparameterlist[best_hyperparams] ,trainData, sess, graphModel)



        testData.groupFolds(trainFolds=[], valFolds=[], testFolds=testFolds)

        acc = validate_mean_over_instances(testData, graphModel, sess, "test")

        print("Final Result:")
        print(acc)











