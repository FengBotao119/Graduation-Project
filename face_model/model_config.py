from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


MODELS = {"svm":[SVC(),{'gamma': [0.001, 0.01, 0.1, 1], 'C':[0.001, 0.01, 0.1, 1,10]}],\
          "logistic":[LogisticRegression(solver='liblinear'),{'penalty':['l1', 'l2']}],\
          "nn":[],\
          "GBDT":[],\
          "xgboost":[],\
          "lightGBM":[]}

MODEL_NAMES = ["svm","logistic","nn","GBDT","xgboost","lightGBM"]