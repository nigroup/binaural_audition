#from cnnmodel import *
import socket
from readData import *
from settings import *
from model import *



trainData = DataSet(trainDir,frames=framelength,folds=trainFolds, overlapSampleSize=25, batchsize=10,shortload=shortload)
testData = DataSet(testDir,frames=framelength,folds=testFolds, overlapSampleSize=25, batchsize=None,shortload=shortload)


# hyperparams
##ratemap
ratemap_ksize = np.array([3, 3])  # todo: hyperparameter?
nr_conv_layers_ratemap = np.array([3, 4])  # four not so good -> maxpooling reduces to fast
sequence_ratemap_pool_window_size = np.array([[1, 2, 2, 1], [1, 3, 3, 1], [1, 2, 3, 1], [1, 2, 2,
                                                                                         1]])  # this is a sequence; build others; first and last repeating... - 1 = batch_size // 1 = feature_map
sequence_ratemap_pool_strides = np.array(
    [[1, 2, 3, 1], [1, 2, 2, 1], [1, 2, 2, 1], [1, 1, 1, 1]])  # todo:hyperparameter?

##ams
ams_ksize = np.array([3, 3, 3])  # todo: hyperparameter?
nr_conv_layers_ams = np.array([3, 4])
sequence_ams_pool_window_size = np.array([[1, 2, 2, 1], [1, 3, 3, 1], [1, 2, 3, 1], [1, 2, 3,
                                                                                     1]])  # this is a sequence; build others; first and last repeating... - 1 = batch_size AND  feature_map - max 4d possible
sequence_ams_pool_strides = np.array([[1, 2, 3, 1], [1, 1, 2, 1], [1, 1, 1, 1], [1, 1, 1, 1]])  # todo:hyperparameter?

# (bs*channel)* cf* time*mf

##both
feature_maps_layer = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])  # all combinations for all layers

##other
number_fully_connected_layers = 3
epochs_per_k_fold_cross_validation = 500


if socket.gethostname()=="eltanin":
    epochs_per_k_fold_cross_validation = 500

else:
    epochs_per_k_fold_cross_validation = 2


hyperparams = {
    "nr_conv_layers_ratemap" : nr_conv_layers_ratemap[0],
    "sequence_ratemap_pool_window_size" : sequence_ratemap_pool_window_size,
    "nr_conv_layers_ams" : nr_conv_layers_ams[0],
    "sequence_ams_pool_window_size" : sequence_ams_pool_window_size,
    "feature_maps_layer" : feature_maps_layer[0:1],
    "epochs_per_k_fold_cross_validation" : epochs_per_k_fold_cross_validation,
    "ams_ksize" : ams_ksize,
    "sequence_ams_pool_strides" : sequence_ams_pool_strides,
    "ratemap_ksize": ratemap_ksize,
    "sequence_ratemap_pool_strides": sequence_ratemap_pool_strides
}

hyperparameterlist = [
    hyperparams, hyperparams
]



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
            print("one fold ended")
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

        print(acc_instance)

        acc = acc + acc_instance

    average_acc_batch = acc / len(data.valX)
    return average_acc_batch






def train(hyperparams ,data, sess,graphModel):
    bestacc = 0

    data.getTrainBatchSize()

    for epoch in range(hyperparams["epochs_per_k_fold_cross_validation"]):

        # training on all apart from k
        for i in range(data.batches):

            train_x, train_y = data.get_next_train_batch()
            sess.run([graphModel.optimiser], feed_dict={graphModel.x: train_x, graphModel.y: train_y, graphModel.cross_entropy_class_weights : data.cross_entropy_class_weights})


            #validation #
            if i%5==0 and len(data.valFolds)>0:
                acc = validate_mean_over_instances(data, graphModel, sess, "val")

                if acc > bestacc:
                    bestacc = acc

    return bestacc



hp_acc = []
for hp_index, hyperparams in enumerate(hyperparameterlist):

    print("train hyperparameter configuration" + str(hp_index))
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

        train(hyperparameterlist[best_hyperparams] ,trainData, sess, graphModel)

        testData.groupFolds(trainFolds=[], valFolds=[], testFolds=testFolds)

        acc = validate_mean_over_instances(testData, graphModel, sess, "test")

        print("Final Result:")
        print(acc)











