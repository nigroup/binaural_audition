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


if socket.gethostname()=="eltanin":
    dir = '/mnt/raid/data/ni/twoears/scenes2018/'
else:
    dir = '/Users/alessandroschneider/Desktop/TuBerlin/bachelor/data_good_short/'


train_dir = dir + "/train"
test_dir = dir + "/test"