import os
import config
config.init()
BASE_DIR = 'E:/'

TRAIN_SET_NAME = 'train_split_Depression_AVEC2017.csv'
DEL_SET_NAME = 'dev_split_Depression_AVEC2017.csv'
TEST_SET_NAME  = 'full_test_split.csv'

TRAIN_ROS_NAME = 'train_ROS.csv'

FACE_HOG = 'hog.csv'

readDepressionWord=open(config.data_dir+'depression_words.txt')
depressionWords = readDepressionWord.read()
depressionWords= [word.strip().lower() for word in depressionWords.split(',')]
readDepressionWord.close()

# COVAREP's clomuns
COVAREP_COLUMNS = ['F0', 'VUV', 'NAQ', 'QOQ', 'H1H2', 'HRF', 'PSP', 'MDQ', 'peakSlope',
'Rd', 'Rd_conf']
for i in range(25): COVAREP_COLUMNS.append('MCEP_' + str(i))
for i in range(25): COVAREP_COLUMNS.append('HMPDM_' + str(i))
for i in range(13): COVAREP_COLUMNS.append('HMPDD_' + str(i))

"""
F0 - Fundamental Frequnency; 原始的声带振动频率，决定了声音的初始音高；一般是在语谱图中最低的共振峰；
VUV - 0,1 表示声音的有无
NAQ - Normalized amplitude quotient
QOQ - quasi-open quotient
-------------以下是频域
H1H2 - the difference in amplitude of the first two harmonics of the differentiated glottal source spectrum 
HRF - Harmonic richness factor
PSP - Parabolic spectral parameter
MDQ - The Maxima Dispersion Quotient (MDQ) quantifies how impulse-like the 
       glottal excitation is through wavelet analysis of the Linear Prediction(LP) residual
peakSlope - A parameter which is essentially a correlate of spectral tilt, derived
            following wavelet analysis. This parameter is effective at discriminating
            lax-tense phonation types
---------- 主要针对LF model
Rd -  estimation of the LF glottal model using Mean Squared Phase (MSP)
Rd_conf -  a confidence value between 0 (lowest confidence) to 1 (best confidence). 
            This last value describes how well the glottal model fits the signal.
---------- HM 谐波模型 主要研究谐波的相位表示
HM PDD - Phase Distortion 导数
HM PDM - Phase Distortion Mean
"""
#读取每个数据所在文件夹前的名称
PREFIX = [folder[:-1] for folder in os.listdir(config.data_dir) \
                                  if folder.endswith('P')]
SUFFIX = {
    'wav': 'AUDIO.wav',
    'face_3d': 'CLNF_features3D.txt',
    'face_2d': 'CLNF_features.txt',
    'gaze': 'CLNF_gaze.txt',
    'pose': 'CLNF_pose.txt',
    'formant': 'FORMANT.csv',
    'text': 'TRANSCRIPT.csv',
    'au': 'CLNF_AUs.txt',
    'hog': 'CLNF_hog.bin',
    'covarep': 'COVAREP.csv'
}

# formant.csv columns
FORMANT_COLUMNS = ['formant_0', 'formant_1', 'formant_2', 'formant_3', 'formant_4']
# column name of video
POSE_COLUMNS = ['Tx', 'Ty',	'Tz', 'Rx', 'Ry', 'Rz']
EXP1_FACE_COLUMNS = ['right_eye_h', 'left_eye_h', 'left_eye_v', 'right_eye_v',
                'mouth_v', 'mouth_h', 'eyebrow_h', 'eyebrow_v']
#reomve 48-67 37,38,43,44 which is not stable 
#44 stable points
X = [ 'x'+str(i) for i in range(68)]
Y = [ 'y'+str(i) for i in range(68)]
removes = [i for i in range(48,68)]+[37,38,43,44]
xRemoves = ['x'+str(i) for i in removes]
yRemoves = ['y'+str(i) for i in removes]
for x,y in zip(xRemoves,yRemoves):
    X.remove(x)
    Y.remove(y)
STABLE_POINTS = X+Y

rightX = ['x'+str(i) for i in range(36,42)]+['x'+str(i) for i in range(17,22)]
rightY = ['y'+str(i) for i in range(36,42)]+['y'+str(i) for i in range(17,22)]
RIGHTEYES = rightX+rightY

leftX = ['x'+str(i) for i in range(42,46)]+['x'+str(i) for i in range(22,27)]
leftY = ['y'+str(i) for i in range(42,46)]+['y'+str(i) for i in range(22,27)]
LEFT_EYES = leftX+leftY

mouthX = ['x'+str(i) for i in range(48,68)]
mouthY = ['y'+str(i) for i in range(48,68)]
MOUTH = mouthX+mouthY

#column name of text:
TEXT_COLUMNS=['sentencesNum','wordsNum','laughters/words','depressions/words',\
    'afinnMean','afinnMedian','afinnMin','afinnMax','afinnStd']

readDepressionWord=open('E:/data/depression_words.txt')
depressionWords = readDepressionWord.read()
depressionWords= [word.strip().lower() for word in depressionWords.split(',')]
readDepressionWord.close()

# feature's name 
FEATURE_EXP_1 = 'exp1'
FEATURE_EXP_2 = 'exp2'

FEATURE_EXP_3 = 'exp3'
FEATURE_EXP_3_VEDIO = 'exp3_vedio'
FEATURE_EXP_3_AUDIO = 'exp3_audio'
FEATURE_EXP_3_TEXT = 'exp3_text'
FEATURE_EXP_3_HOGPCA = 'exp3_hog'

FEATURE_FINAL_COVAREP = 'final_covarep'
FEATURE_FINAL_FORMANT = 'final_formant'
FEATURE_FINAL_FAUs = 'final_faus'
FEATURE_FINAL_GAZE_POSE = 'final_gaze_pose'
FEATURE_FINAL_TEXT = 'final_text'
FEATURE_FINAL_VGG = 'final_vgg'
FEATURE_FINAL_ALEXNET = 'final_alexnet'
FEATURE_FINAL_FAUs_BOW = 'final_faus_bow'
FEATURE_FINAL_GAZE_POSE_BOW = 'final_gaze_pose_bow'
FEATURE_FINAL_COVAREP_BOW = 'final_covarep_bow'
FEATURE_FINAL_FORMANT_BOW = 'final_formant_bow'
FEATURE_FINAL_FUSION = 'final_fusion'

FEATURE_BL = 'baseline'
FEATURE_FINAL ='final'



# model's name
MODEL_RF = 'rf'
MODEL_RNN ='rnn'
MODEL_LINEAR = 'lr'

AUDIO_TABLE = set([
    config.tbl_exp2_audio_fea,
    config.tbl_exp3_audio_fea
])

VIDEO_TABLE = set([
    config.tbl_exp1_face_fea,
    config.tbl_exp1_head_fea,
    config.tbl_exp3_hog_fea,
    config.tbl_exp3_vedio_fea
])

TEXT_TABLE = set([config.tbl_exp3_text_fea])
