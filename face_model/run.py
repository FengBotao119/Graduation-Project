from sklearn.datasets import load_iris
from models import Model

"""

"""

def Train(model,X,Y,n_folds,out_dir,file_name):
    model.gridsearch(X,Y,n_folds)
    model.save(out_dir,file_name)
    return model,model.get_best_param,model.get_best_score

def Evaluate(model,X,Y):
    return model.evaluate(X,Y)


model = Model('logistic')

Data = load_iris()
X = Data.data[:100]
Y = Data.target[:100]

model,best_param,best_score = Train(model,X,Y,5,'./face_model',"svm.model")
print(best_param,best_score)

print(Evaluate(model,X,Y))

