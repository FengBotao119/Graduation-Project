from global_values import *

class FeatureExtration():
    def __init__(self, feature):
        self.feature = feature

    def gen_fea(self):
        if self.feature == FEATURE_EXP_2:
            from core.feature_extraction.example_2 import gen_feature
            gen_feature.gen_fea()
        elif self.feature == FEATURE_EXP_1:
            from core.feature_extraction.example_1 import gen_feature
            gen_feature.gen_fea()
        elif self.feature == FEATURE_EXP_3:
            from core.feature_extraction.example_3 import gen_feature
            gen_feature.gen_fea()
        elif self.feature == FEATURE_FINAL:
            from core.feature_extraction.final_model import gen_feature
            gen_feature.gen_fea()
        else:
            print(self.feature, 'not finished yet!')