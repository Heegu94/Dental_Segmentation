import glob
import numpy as np
from tensorflow.keras.utils import to_categorical

def gen(data_dir, batch):
    
    idx = 0
    x = np.load(sorted(glob.glob(data_dir+'x_*.npy'))[0])
    y = np.load(sorted(glob.glob(data_dir+'y_*.npy'))[0])
    
    while 1:
        idx_list = list(range(0, len(x)))
        np.random.shuffle(idx_list) #
        bat_img = []
        bat_lab = []
        
        #
        if idx >= len(idx_list) - batch:
            tmp_list = idx_list[idx:]
            idx = 0
        else:
            tmp_list = idx_list[idx:idx+batch]
            idx = idx + batch
        
        # 
        for i in tmp_list:
            x_ = x[i]
            y_ = y[i]

            if np.random.random()>0.5:
                x_tmp = np.fliplr(x_)
                y_tmp = np.fliplr(y_)
                
                bat_img.append(x_tmp)
                bat_lab.append(y_tmp)
            else : 
                bat_img.append(x_)
                bat_lab.append(y_)
                
                            
        yield np.array(bat_img), np.array(bat_lab)