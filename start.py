import argparse
from core.feature_extraction.FeaturExtraction import FeaturExtraction
from core.model.Train import Train

def cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',help="feature extraction or training",type=str,choices=['extraction','train'])
    parser.add_argument('--feature',help="choose which feature to extract or train",type=str)
    parser.add_argument('--model',help="choose which model to train",type=str)
    parser.add_argument('--feature_table',help='choose the feature to train model using table name',nargs='+')

    args = parser.parse_args()

    if args.mode == "extraction":
        print(f"You are extracting feature {args.feature}!")
        extractor = FeaturExtraction(args.feature)
        extractor.gen_fea()

    elif args.mode == "train":
        if args.feature:
            print(f"You are training using model {args.model} via feature {args.feature}!")
            model = Train.train(args,feature,args.model)
            model.train()

        else:
            print(f"You are training using model {args.model} via feature {args.feature_table}!")


if __name__=='__main__':
    cmd()
