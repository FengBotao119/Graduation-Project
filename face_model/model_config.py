from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

MODELS = {"svm":(SVC(probability=True,random_state=123),{'kernel':['linear','poly','rbf','sigmoid'], 'C':[0.001, 0.1, 10]}),\
          "randomforest":(RandomForestClassifier(random_state=123),{'n_estimators':[50,100,150,200,250,300],'criterion':['gini','entropy'],'max_depth':[5,10,15,20,30]})}


MODEL_NAMES = ["svm",'randomforest']
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