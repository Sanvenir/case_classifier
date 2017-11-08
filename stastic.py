#!/usr/bin/python
# -*- coding: UTF-8 -*-

from collections import defaultdict
import json

from train import Classifier


if __name__ == "__main__":
    train_data = Classifier.read_train_file("BDCI2017-minglue/1-train/train.txt")
    penalty_dict = defaultdict(int)
    for elements in train_data:
        penalty_dict[elements[2]] += 1
    print(penalty_dict)

    result_dict = defaultdict(int)
    with open('data.json') as f:
        while True:
            line = f.readline()
            if not line:
                break
            result_dict[json.loads(line)["penalty"]] += 1
    print(result_dict)
