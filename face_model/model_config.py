from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


MODELS = {"svm":(SVC(probability=True),{'gamma': [0.001, 0.01, 0.1, 1], 'C':[0.001, 0.1, 10]})}
          #"logistic":(LogisticRegression(solver='liblinear'),{'penalty':['l1', 'l2']})}
          #"nn":(),\
          #"GBDT":(),\
          #"xgboost":(),\
          #"lightGBM":()}

MODEL_NAMES = ["svm"]
#["svm","logistic","nn","GBDT","xgboost","lightGBM"]

EXPRESSION_LABEL = {"0":"angry",
                    "1":"disgust",
                    "2":"fear",
                    "3":"happy",
                    "4":"sad",
                    "5":"surprise",
                    "6":"neutral"}

FEATURE_COLUMNS = ['confidence', ' AU01_r', ' AU02_r',
        ' AU04_r', ' AU05_r', ' AU06_r', ' AU07_r', ' AU09_r', ' AU10_r',
        ' AU12_r', ' AU14_r', ' AU15_r', ' AU17_r', ' AU20_r', ' AU23_r',
        ' AU25_r', ' AU26_r', ' AU45_r', ' AU01_c', ' AU02_c', ' AU04_c',
        ' AU05_c', ' AU06_c', ' AU07_c', ' AU09_c', ' AU10_c', ' AU12_c',
        ' AU14_c', ' AU15_c', ' AU17_c', ' AU20_c', ' AU23_c', ' AU25_c',
        ' AU26_c', ' AU28_c', ' AU45_c',]