import argparse
import json
import math
import random

from evaluate_model import evaluateModel

parser = argparse.ArgumentParser(description='MultiWoz Eval Script')
parser.add_argument('--target', type=str, default='target file')
parser.add_argument('--pred', type=str, default='pred file')
args = parser.parse_args()

predictions = json.load(open(args.pred))

predictions = { k:predictions[k] for k in predictions.keys()}

targets = json.load(open(args.target))

evaluateModel(predictions, targets, mode='test')
