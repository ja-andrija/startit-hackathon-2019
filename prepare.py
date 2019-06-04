import sys
import argparse
import os
import shutil

if __name__=="__main__":
    parser=argparse.ArgumentParser()

    parser.add_argument('--dataset', '-d', help='Dataset folder path with train.json and test.json files', type=str)
    parser.add_argument('--train_test_ratio', '-r', help='Train to test ratio used for split', type=float, default=0.8)
    
    args = parser.parse_args()

    dataset_path = args.dataset
    split_ratio = args.train_test_ratio

    # Copy test.json to cache/test.json
    shutil.copy2(os.path.join(dataset_path, "test.json"), "/cache/test.json")

    # Split train.json into train_split.json and test_split.json