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
            #also calc standardize matrix here

            acc,model = train(hyperparams,data,sess,graphModel)
            avg_acc= acc + avg_acc
            print("one fold ended")
    return avg_acc/k


def mylog(y,y_,acc):

    pdb.set_trace()

    pass


def train(hyperparams ,data, sess,graphModel):
    bestmodel = "bestesModell" #?
    bestacc = 0

    data.getTrainBatchSize()

    for epoch in range(hyperparams["epochs_per_k_fold_cross_validation"]):

        # training on all apart from k
        for i in range(data.batches):

            train_x, train_y = data.get_next_train_batch()
            sess.run([graphModel.optimiser], feed_dict={graphModel.x: train_x, graphModel.y: train_y, graphModel.cross_entropy_class_weights : data.cross_entropy_class_weights})



            if i%5==0:

                val_x, val_y = data.getData("val")
                o_recall = sess.run([graphModel.recall], feed_dict={ graphModel.y:val_y, graphModel.x: val_x })
                acc = o_recall

                print(acc)

                if acc > bestacc:
                    bestacc = acc


                    #tf.train.Saver().save(sess, 'my_test_model')
                    #tf.train.Saver -- bestimmte Variablen (Klasse:Modell )


    return bestacc, bestmodel


for hp_index, hyperparams in enumerate(hyperparameterlist):

    print("train hyperparameter configuration" + str(hp_index))
    with tf.Graph().as_default() as g:
        graphModel = GraphModel(hyperparams)
        kfold(hyperparams, trainData, graphModel, g)





print("Ende")




'''
testData = DataSet(testDir,frames=framelength,folds=[7,8], batchsize=None,shortload=shortload,model=model)
'''





'''
for para_conf in hyperparameters_configurations:
    avg_acc[param_conf] = kfold(param_conf)


    #choose best loop and save hyperparams

'''



#trainData.groupFolds(trainFolds=[1,2,3,4,5],valFolds=[],testFolds=[])
#train(hyperparams, trainData)



#hier testen - ergebnis fuer paper




