from global_values import *

class FeaturExtraction():
    def __init__(self,feature):
        self.feature = feature
    
    def gen_fea(self):
        if self.feature == 'text':
            pass
        elif self.feature == 'video':
            pass
        elif self.feature == 'audio':
            pass
        else:
            print('This feature does not exist!')