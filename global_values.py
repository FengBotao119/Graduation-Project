import os


TRAIN_DATASET = "train_split_Depression_AVEC2017.csv"
DEV_DATASET = "dev_split_Depression_AVEC2017.csv"
TEST_DATASET = "full_test_split.csv"
DATA_DIR = "./data/"
TEXT_DIR = "./data/text/"

GRAD_CLIP = 5

PREFIX = sorted([file[:3] for file in os.listdir(TEXT_DIR) if file.endswith('.csv')])
