import socket

#run dependend Settings
model = "framewise_cnn"

local = True

if socket.gethostname()=="eltanin":
    local = False
    dir = '/mnt/raid/data/ni/twoears/scenes2018/'
    save_path = '/home/alessandroschneider/log'
else:
    dir = '/Users/alessandroschneider/Desktop/TuBerlin/bachelor/data_good_short/'
    save_path = '/Users/alessandroschneider/Desktop/TuBerlin/bachelor/log'



#Test Settings
check_rejected_data = True

#Data Settins
n_features = 160
n_ratemap_features = 32
n_ams_features = 128
n_ams_features_cf = 16
n_ams_features_mf = 8
n_labels = 13


logs_path="./log/cnn"

timesteps = 49



#Heuristic Settings LDL
ldl_n_batches = 500
ldl_timesteps = 49
ldl_overlap = 25
ldl_buffer_rows = 16 #6,16,32 - reijectino rate


if local==True:
    ldl_n_batches = 10
    ldl_buffer_rows = 4


train_dir = dir + "/train"
test_dir = dir + "/test"



#Util function
def log(file, text):
    import sys
    log_file = open(file, "a")
    old_stdout = sys.stdout
    sys.stdout = log_file
    print(text)
    sys.stdout = old_stdout
    log_file.close()


