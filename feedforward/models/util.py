import os, shutil
import numpy as np
import pdb

def clear_folder(folder):

    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
                print(file_path + " deleted")
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)
    print("folder cleared")



def showFull():
    print("tset")
    np.set_printoptions(threshold=np.nan)


''' calculate confusion matrix '''
def confusion_matrix(y,y_):
    y_ = y_*2
    correct_classes = np.count_nonzero((y+y_)==3,axis=0)
    total_labels = np.count_nonzero(y==1,axis=0)

    print correct_classes
    print total_labels 
    
def test():
    print("ok")


def sig(x, derivative=False):
  return x*(1-x) if derivative else 1/(1+np.exp(-x))
    
   
def crossentropy(oy,oy_,weights=None):
    if weights==None:
        weights=np.ones(oy.shape[1])
    print("ok")
    pdb.set_trace()
    ersterSummand =weights*oy * -np.log(sig(oy_)) 
    zweiterSummand = (1-oy)*-np.log(1-sig(oy_))
    pdb.set_trace()
    return ersterSummand +  zweiterSummand

    
    
    
    
    
