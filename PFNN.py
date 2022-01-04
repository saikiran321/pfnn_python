import os
import sys
import math
import numpy as np
import time
M_PI = np.pi


# XDIM = 468
# YDIM = 376
# HDIM = 512

# XDIM = 342
# YDIM = 311
# HDIM = 512

weight_path="/Users/jaya.patibandla/Desktop/jio/AI4Animation/AI4Animation/AI4Animation/SIGGRAPH_2017/TensorFlow/trained_Adam/"
# weight_path="/Users/jaya.patibandla/Desktop/jio/smpl/animate/pfnn_easy_to_use/assets/network/pfnn/"

def load_weights(wpath):
    
    data = np.fromfile(wpath,dtype='float32')

    return data

def ELU(z):
    return np.where(z > 0, z, 1 * (np.exp(z) - 1))


class PFNN:


    def __init__(self,weight_path,XDIM=342,YDIM=311,HDIM=512,Phase_index=3) -> None:
            
        self.W0p =  np.zeros((50,HDIM, XDIM))
        self.W1p = np.zeros((50,HDIM, HDIM))
        self.W2p = np.zeros((50,YDIM, HDIM))
        
        self.b0p = np.zeros((50,HDIM))
        self.b1p = np.zeros((50,HDIM))
        self.b2p = np.zeros((50,YDIM))

        self.phase = 0
        self.damping = 0
        self.phase_index = Phase_index
        

        self.Xmean = load_weights(weight_path+'Xmean.bin')
        self.Xstd = load_weights(weight_path+'Xstd.bin')
        self.Ymean = load_weights(weight_path+'Ymean.bin')
        self.Ystd = load_weights(weight_path+'Ystd.bin')




        for i in range(50):
            self.W0p[i] = load_weights( weight_path+"W0_{:03d}.bin".format(i)).reshape(HDIM,XDIM)
            self.W1p[i] =load_weights( weight_path+"W1_{:03d}.bin".format(i)).reshape(HDIM,HDIM)
            self.W2p[i] =load_weights( weight_path+"W2_{:03d}.bin".format(i)).reshape(YDIM,HDIM)
            self.b0p[i] =load_weights( weight_path+"b0_{:03d}.bin".format(i)).reshape(HDIM)
            self.b1p[i] = load_weights( weight_path+"b1_{:03d}.bin".format(i)).reshape(HDIM)
            self.b2p[i] = load_weights( weight_path+"b2_{:03d}.bin".format(i)).reshape(YDIM)  
            
        print("sucessfully loaded")


    def setdamping(self,val):
        self.damping  = val


    def predict(self,Xp):

        pindex_1 = int((self.phase/(2*np.pi))*50)
    
        Xp = (Xp - self.Xmean) / self.Xstd


        H0 = (self.W0p[pindex_1] @ Xp) + self.b0p[pindex_1]
        H0 = ELU(H0)

        H1 = (self.W1p[pindex_1] @ H0) + self.b1p[pindex_1]
        H1 = ELU(H1)
        Yp = (self.W2p[pindex_1]@ H1) + self.b2p[pindex_1]

        Yp = (Yp * self.Ystd) + self.Ymean


        self.phase = math.fmod(self.phase+(1-self.damping)*2*np.pi * Yp[self.phase_index],2*np.pi)


        return Yp



if __name__=="__main__":
    XDIM = 468
    YDIM = 376
    HDIM = 512

    pfnn =PFNN(weight_path,XDIM,YDIM,HDIM)




    xp = np.random.random(XDIM)
    phase = np.sin(np.random.random()*M_PI)

    start_time = time.time()
    yp = pfnn.predict(xp)
    end_time = time.time()
    print(end_time-start_time)







