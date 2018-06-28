import socket


#Data Settins
n_features = 160
n_ratemap_features = 32
n_ams_features = 128
n_ams_features_cf = 16
n_ams_features_mf = 8
n_labels = 13


logs_path="./log/cnn"

#run dependend Settings
model = "framewise_cnn"


testFolds =  [7,8]
trainFolds = [1,2,3,4,5,6]

if socket.gethostname()=="eltanin":
    trainDir = '/mnt/raid/data/ni/twoears/scenes2018/train'
    testDir = '/mnt/raid/data/ni/twoears/scenes2018/test'

else:
    trainDir = '/Users/alessandroschneider/Desktop/TuBerlin/bachelor/data/train/'
    testDir = '/Users/alessandroschneider/Desktop/TuBerlin/bachelor/data/test'