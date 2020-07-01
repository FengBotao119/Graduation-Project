import numpy as np 
import config
import librosa
import librosa.display
import matplotlib.pyplot as plt

#用于LLD特征的提取
def slideWindows(slide_size,hop_size,fps,data):
    #对于不同的数据需要更改读取方式
    data = data
    max_len_seq = int(data.shape[0]/hop_size/fps) 
    num_llds = data.shape[1]

    X_func = np.zeros((max_len_seq,num_llds*2))
    window_size_half = int(slide_size*fps/2)
    time_stamps_new = np.zeros((max_len_seq,1))

    for t in range(0,max_len_seq):
        t_orig = t*fps*hop_size
        min_orig = max(0,t_orig-window_size_half)
        max_orig = min(data.shape[0],t_orig+window_size_half+1)
        if min_orig<max_orig and t_orig<=data.shape[0]:
            time_stamps_new[t] = t*hop_size
            X_func[t,:num_llds] = np.mean(data[min_orig:max_orig,:],axis=0)
            X_func[t,num_llds:] = np.std(data[min_orig:max_orig,:],axis=0)
        else:
            time_stamps_new = time_stamps_new[:t,:]
            X_func = X_func[:t,:]
            break

    X_func = np.concatenate((time_stamps_new,X_func),axis=1)
    return X_func

def slideWindowsDeep(slide_size,hop_size,fps,data,file):
    #用于Deep Spectrum图片的提取的提取
    max_len_seq = int(data.shape[0]/hop_size/fps)+1
    window_size = slide_size*fps

    for t in range(0,max_len_seq):
        t_orig = t*fps*hop_size
        min_orig = t_orig
        max_orig = min(data.shape[0],t_orig+window_size+1)
        if min_orig<max_orig and t_orig<=data.shape[0]:
            melspec = librosa.feature.melspectrogram(data[min_orig:max_orig], fps, n_fft=512, hop_length=256, n_mels=128)
            logmelspec = librosa.power_to_db(melspec)
            plt.figure(figsize=(0.97,0.98))
            librosa.display.specshow(logmelspec,sr=fps,cmap='viridis')
            plt.axis('off')
            plt.savefig(config.data_dir+file+'P/'+'/img/'+str(min_orig/fps)+'-'+str(round(max_orig/fps,1))+'.jpg',bbox_inches='tight',dpi=298,pad_inches=0.0)
            plt.close()
        else:
            break



