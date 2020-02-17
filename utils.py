import os, cv2, tqdm
import numpy as np

def load_3d_vol(img_paths, mode = 'gray'):
    '''
    ======== Input ==========
    img_paths (list type): Path of images
    ex) ['/test/test/001.bmp', '/test/test/002.bmp', ..., '/test/test/00n.bmp']
    '''
    
    cmap_dict = {
                'gray': cv2.COLOR_BGR2GRAY,
                'rgb' : cv2.COLOR_BGR2RGB
                }
    output = [cv2.cvtColor(cv2.imread(img_path), cmap_dict[mode]) for img_path in img_paths]

    return np.array(output)

def make_mask(img3d):
    '''
    ======== Input ==========
    img3d (3d numpy array): Scan of One Patient [Slice, Height, Width] 
    '''
    
    hsv = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[...,0] for img in img3d]) # img3d로 들어온 배열(147, 512, 512, 3) 을 하나씩 뽑는다 img 배열 : 

    s, h, w = hsv.shape # hsv.shape = (147, 512, 512)
    output = np.ones((s, h, w, 3)) # output.shape = (147,512,512,3)
    output[...,1:] = 0 # output의 2,3 채널은 전부 0
    # hsv[.... 256].shape = (147, 512,) -> rot90([], 2) -> 90 도씩 두번 회전; (hsv[.... 256] = 147,512,512 의 image에서 중앙값들만 추출)
    center = np.rot90(hsv[..., 256], 2) 
    tmp = np.max(center, 1) # center.shape = (147, 512,) => max([ ], 1) => 각 slice 별 max값 추출. => tmp.shape = (147, )
    mn_range = [tmp[tmp!=0][-1]-2, tmp[tmp!=0][-1]+2] # tmp[tmp!=0][-1] => tmp 리스트에 0을 뺀 나머지 중 마지막 인자 출력
    s, h, w = np.where(np.logical_and(hsv>=mn_range[0], hsv<=mn_range[1]))
    output[s, h, w, 1] = 1 # 1 채널에 1 : mn
    output[s, h, w, 0] = 0 # 0 채널에 0
    
    mx_range = [tmp[tmp!=0][0]-2, tmp[tmp!=0][0]+2]
    s, h, w= np.where(np.logical_and(hsv>=mx_range[0], hsv<=mx_range[1]))
    output[s, h, w, 2] = 1 # 2 채널에 1 : mx
    output[s, h, w, 0] = 0 # 0 채널에 0
    
    return output


# IMAGE : RGB to GRAY
# MASK : MNX to HSV

class DataLoader_2d():
    def __init__(self, path, n_val):
        """
        path : Data Path that have directory of data, label
        """
        self.root = path
        self.total_list = [i for i in sorted(os.listdir(os.path.join(self.root, 'data'))) if '.npy' in i]
        self.train_list = [i for i in self.total_list[:-n_val]]
        self.val_list = [i for i in self.total_list[-n_val:]]
    
    def TrainGenerator(self, patch_size, batch_size):
        while 1:
            for path in self.train_list:
                tmp_data = np.expand_dims(np.load(os.path.join(self.root, 'data', path)), -1)
                tmp_lab = np.load(os.path.join(self.root, 'label', path))/255.
                
                to_slice = tmp_lab.argmax(axis=-1)
                to_slice = to_slice.sum(axis=(1, 2))
                
                tmp = tmp_lab[..., 1:].copy()
                tmp = tmp.sum(axis=(0, -1))

                row, col = np.where(tmp!=0)

                row_idx = [row.min(), (row.min() + row.max()-patch_size)//2, row.max()-patch_size]
                col_idx = [col.min(), (col.min() + col.max()-patch_size)//2, col.max()-patch_size]

                cnt = 0
                aug_data = None
                aug_lab = None
                for idx in row_idx:
                    for jdx in col_idx:
                        if cnt ==0:
                            aug_data = tmp_data[to_slice!=0][:, idx:idx+patch_size, jdx:jdx+patch_size]
                            aug_lab = tmp_lab[to_slice!=0][:, idx:idx+patch_size, jdx:jdx+patch_size]
                            #print(aug_data.shape)
                        else:
                            #print(tmp_data[:, idx:idx+patch_size, jdx:jdx+patch_size].shape)
                            aug_data = np.concatenate((aug_data, 
                                                       tmp_data[to_slice!=0][:, idx:idx+patch_size, jdx:jdx+patch_size]), axis=0)
                            aug_lab = np.concatenate((aug_lab, 
                                                      tmp_lab[to_slice!=0][:, idx:idx+patch_size, jdx:jdx+patch_size]), axis=0)
                        cnt +=1
                vmax = aug_data.max(axis=(1, 2, 3), keepdims=True)
                vmax[vmax==0]=1
                aug_data = aug_data/vmax
                tr_idx = np.random.choice(len(aug_data), size = len(aug_data), replace=False)
                for i in range(0, len(tr_idx), batch_size):
                    yield aug_data[tr_idx[i:i+batch_size]], aug_lab[tr_idx[i:i+batch_size]]
                    
    def ValidationGenerator(self, patch_size, batch_size):
        while 1:
            for path in self.val_list:
                tmp_data = np.expand_dims(np.load(os.path.join(self.root, 'data', path)), -1)
                tmp_lab = np.load(os.path.join(self.root, 'label', path))/255.
                
                to_slice = tmp_lab.argmax(axis=-1)
                to_slice = to_slice.sum(axis=(1, 2))
                
                tmp = tmp_lab[..., 1:].copy()
                tmp = tmp.sum(axis=(0, -1))

                row, col = np.where(tmp!=0)

                row_idx = [row.min(), (row.min() + row.max()-patch_size)//2, row.max()-patch_size]
                col_idx = [col.min(), (col.min() + col.max()-patch_size)//2, col.max()-patch_size]

                cnt = 0
                aug_data = None
                aug_lab = None
                for idx in row_idx:
                    for jdx in col_idx:
                        if cnt ==0:
                            aug_data = tmp_data[to_slice!=0][:, idx:idx+patch_size, jdx:jdx+patch_size]
                            aug_lab = tmp_lab[to_slice!=0][:, idx:idx+patch_size, jdx:jdx+patch_size]
                            #print(aug_data.shape)
                        else:
                            #print(tmp_data[:, idx:idx+patch_size, jdx:jdx+patch_size].shape)
                            aug_data = np.concatenate((aug_data, 
                                                       tmp_data[to_slice!=0][:, idx:idx+patch_size, jdx:jdx+patch_size]), axis=0)
                            aug_lab = np.concatenate((aug_lab, 
                                                      tmp_lab[to_slice!=0][:, idx:idx+patch_size, jdx:jdx+patch_size]), axis=0)
                        cnt +=1
                vmax = aug_data.max(axis=(1, 2, 3), keepdims=True)
                vmax[vmax==0]=1
                aug_data = aug_data/vmax
                tr_idx = np.random.choice(len(aug_data), size = len(aug_data), replace=False)
                for i in range(0, len(tr_idx), batch_size):
                    yield aug_data[tr_idx[i:i+batch_size]], aug_lab[tr_idx[i:i+batch_size]]
                
class DataLoader_3d():
    def __init__(self, path, n_val):
        self.root = path
        self.total_list = [i for i in sorted(os.listdir(self.root)) if '.DS_Store' != i]
        self.train_list = [os.path.join(self.root, i) for i in self.total_list[:-n_val]]
        self.val_list = [os.path.join(self.root, i) for i in self.total_list[-n_val:]]
        
    def TrainGenerator(self, vox_size, split_stirde, batch):
        while 1:
            for path in self.train_list:
                #print(path)
                Original_PATH = os.path.join(path, 'Original')
                MNX_PATH = os.path.join(path, 'MxMn')

                Ori_list = [i for i in sorted(os.listdir(Original_PATH)) if 'bmp' in i]
                MNX_list = [i for i in sorted(os.listdir(MNX_PATH)) if 'bmp' in i]

                Ori_paths = [os.path.join(Original_PATH, i) for i in Ori_list]
                MNX_paths = [os.path.join(MNX_PATH, i) for i in MNX_list]
                
                #print("Load image")
                raw_data = load_3d_vol(Ori_paths)
                raw_data = np.expand_dims(raw_data, -1)
                
                #print("Load mask")
                raw_mask = load_3d_vol(MNX_paths, mode='rgb')
                raw_mask = make_mask(raw_mask)
                
                s, h, w, _ = raw_data.shape
                
                sdx = np.arange(0, s - vox_size)
                np.random.shuffle(sdx)
                hdx = np.arange(0, h - vox_size, split_stride)
                np.random.shuffle(hdx)
                wdx = np.arange(0, w - vox_size, split_stride)
                np.random.shuffle(wdx)
                
                cnt = 0
                output_data = []
                output_label = []
                for _s in sdx:
                    for _h in hdx:
                        for _w in wdx:
                            if cnt ==0:
                                output_data = []
                                output_label = []
                            output_data.append(raw_data[_s:_s+vox_size, _h:_h+vox_size, _w:_w+vox_size])
                            output_label.append(raw_mask[_s:_s+vox_size, _h:_h+vox_size, _w:_w+vox_size])
                            cnt += 1
                            if len(output_data) == batch:
                                cnt = 0
                                yield np.array(output_data), np.array(output_label)
                                