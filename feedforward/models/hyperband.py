#from cnnmodel import *
import socket
from readData import *
from settings import *
from cnnmodel import *



trainData = DataSet(trainDir,frames=framelength,folds=trainFolds, batchsize=20,shortload=shortload,model="framewise_cnn")
testData = DataSet(testDir,frames=framelength,folds=testFolds, batchsize=None,shortload=shortload,model="framewise_cnn")




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








############################################################
# train:
x = tf.placeholder(tf.float32, shape=(None, n_features, framelength, 1), name="x")  # None=batch_size, 1=channels
y = tf.placeholder(tf.float32, shape=(None, n_labels), name="y")  # (5000, 13)

y_ = model(hyperparams, x)

# cross_entropy = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=y, logits=y_, pos_weight=tf.constant(weights)))
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_)
optimiser = tf.train.AdamOptimizer(learning_rate=0.4).minimize(cross_entropy)
# init_op = tf.global_variables_initializer()

############################################################
# val
proby_ = tf.nn.sigmoid(y_)
sigmoid = tf.nn.sigmoid(y_)

correct_prediction = tf.equal(sigmoid > 0.5, y == 1)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

tf.summary.scalar('accuracy', accuracy)
merged = tf.summary.merge_all()
test_writer = tf.summary.FileWriter(logs_path + "/test" + str(time.time()), graph=tf.get_default_graph())
# balanced accuracy
############################################################
init_op = tf.global_variables_initializer()




def kfold(hyperparams, data, sess):
    avg_acc=0
    for k in trainFolds:
        copytrainFolds = trainFolds[:]
        copytrainFolds.remove(k)
        data.groupFolds(trainFolds=copytrainFolds,valFolds=[k],testFolds=[])
        print(k)
        print(copytrainFolds)
        acc,model = train(hyperparams,data,sess)
        avg_acc= acc + avg_acc
        print("one fold ended")
    return avg_acc/k


def train(hyperparams ,data, sess):


    bestacc = 0

    data.getTrainBatchSize()

    for epoch in range(hyperparams["epochs_per_k_fold_cross_validation"]):

        # training on all apart from k
        for i in range(data.batches):
            train_x, train_y = data.get_next_train_batch()
            sess.run([optimiser], feed_dict={x: train_x, y: train_y})



            if i%5==0:

                val_x,val_y = data.getData("val")
                acc,summary = sess.run([accuracy,merged], feed_dict={x: val_x, y:val_y })
                print(acc)
                if acc > bestacc:
                    bestacc = acc
                    bestmodel = "bestesModell"



    return bestacc, bestmodel












with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    #loop around hyperparams
    kfold(hyperparams,trainData,sess)



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




