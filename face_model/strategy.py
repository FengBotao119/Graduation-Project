"""
One VS One
One VS Rest
Error-Correcting Output-Codes

input: training set:

OVO output: class1_2_trainset, class1_3_trainset...
OVR output: class1_trainset, class2_trainset, class3_trainset...

"""

def OVR(data):
    classes = data.iloc[:,-1].unique()
    for class_ in classes:
        data[str(class_)] = [1 if value==class_ else 0 for value in data.iloc[:,-1]]
    return data


