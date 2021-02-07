import argparse
import csv
import pandas as pd

from extractor.feature import get_feature
from extractor.train import train


def make_features(path):
    with open(path, 'r') as f:
        tsv_reader = csv.reader(f, delimiter='\t')
        data = [dict(url=row[1], path=row[2]) for row in tsv_reader][1:]
    get_feature(data)


def train_model(path):
    df = pd.read_csv(path)
    df = df.fillna({'title_dist': 0.0})
    df['concat_attr_name'] = df['attr_name'].str.cat(df['parent_attr_name'], sep='|')
    train(df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-t', '--task', choices=['feature', 'train'], required=True, help='choose task to apply')
    parser.add_argument('-d', '--data-path', required=True, help='path to data')
    args = parser.parse_args()

    if args.task == 'feature':
        make_features(args.data_path)
    elif args.task == 'train':
        train_model(args.data_path)
