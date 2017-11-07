#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import tensorflow as tf
import numpy as np

DICT_PATH = "word_dict.txt"
LAW_NUM = 453
HIDDEN_LAYER_NUM = 64
TEMP_PATH = "tmp"
SOURCE_FILE_NAME = "source_data"


def read_train_file(file_name):
    if not tf.gfile.Exists(file_name):
        raise FileNotFoundError("Train file '{}' not found".format(file_name))

    result = list()
    with tf.gfile.Open(file_name) as train_file:
        while True:
            line = train_file.readline()
            if not line:
                break
            assert isinstance(line, str)
            elements = line.split("\t")
            laws = elements[3].split(",")
            result.append([int(elements[0]), elements[1], int(elements[2]), [int(law) for law in laws]])

    return result


class LawsClassifier:
    def __init__(self, source_data, testing_percentage, validation_percentage):
        self.word_dict = self.open_dict(DICT_PATH)
        self.training_text_list = []
        self.training_laws_list = []
        self.testing_text_list = []
        self.testing_laws_list = []
        self.validation_text_list = []
        self.validation_laws_list = []

        self.load_data(source_data, testing_percentage, validation_percentage)
        print("Complete.")
        text_placeholder = tf.placeholder(tf.float32, shape=[None, len(self.word_dict)])
        laws_placeholder = tf.placeholder(tf.float32, shape=[None, LAW_NUM])

        text_normalize = tf.nn.l2_normalize(text_placeholder, dim=0)

        l1 = self.add_layer(text_normalize, len(self.word_dict), HIDDEN_LAYER_NUM,
                            activation_function=tf.nn.relu)
        prediction = self.add_layer(l1, HIDDEN_LAYER_NUM, LAW_NUM, activation_function=None)

        loss = tf.reduce_mean(tf.reduce_sum(tf.square(laws_placeholder - prediction)))

        train_step = tf.train.GradientDescentOptimizer(0.000001).minimize(loss)

        init = tf.global_variables_initializer()
        print("Starting session...")
        with tf.Session() as sess:
            sess.run(init)
            for i in range(1000):
                sess.run(train_step, feed_dict={text_placeholder: self.training_text_list,
                                                laws_placeholder: self.training_laws_list})
                print("Training loss: {}".format(sess.run(loss, feed_dict={text_placeholder: self.training_text_list,
                                                                           laws_placeholder: self.training_laws_list})))
                print("Testing loss: {}".format(sess.run(loss, feed_dict={text_placeholder: self.testing_text_list,
                                                                          laws_placeholder: self.testing_laws_list})))

    def load_data(self, source_data, testing_percentage, validation_percentage):
        print("Preparing source data...")
        for i, elements in enumerate(source_data):
            if not i % 1000:
                print("{} / {} complete.".format(i, len(source_data)))
            checker = random.randrange(100)
            text_list = self.processed_string(self.word_dict, elements[1], file_name="tmp/{}.tmp".format(elements[0]))
            laws_list = self.processed_law(elements[3])
            if checker < testing_percentage:
                self.testing_text_list.append(text_list)
                self.testing_laws_list.append(laws_list)
            elif checker < testing_percentage + validation_percentage:
                self.validation_text_list.append(text_list)
                self.validation_laws_list.append(laws_list)
            else:
                self.training_text_list.append(text_list)
                self.training_laws_list.append(laws_list)

    @staticmethod
    def add_layer(inputs, in_size, out_size, activation_function=None):
        weights = tf.Variable(tf.random_normal([in_size, out_size]))
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        wx_plus_b = tf.matmul(inputs, weights) + biases
        if activation_function is None:
            outputs = wx_plus_b
        else:
            outputs = activation_function(wx_plus_b)
        return outputs

    @staticmethod
    def open_dict(dict_path):
        if not tf.gfile.Exists(dict_path):
            raise FileNotFoundError("Dict file '{}' not found!".format(dict_path))
        result = list()
        with tf.gfile.Open(dict_path, 'r+') as dict_file:
            while True:
                line = dict_file.readline()
                if not line:
                    break
                line = line.strip()
                if line and not line.isdigit():
                    result.append(line)
            for word in result:
                dict_file.write("{}\n".format(word))
        return result

    @staticmethod
    def processed_string(words_dict, string, file_name=None):
        if tf.gfile.Exists(file_name):
            return np.fromfile(file_name, dtype=np.float32)
        result = np.array([string.count(word) for word in words_dict], dtype=np.float32)
        if file_name is not None:
            result.tofile(file_name)
        return result

    @staticmethod
    def processed_law(law_list):
        result = [0 for _ in range(453)]
        for law in law_list:
            result[law] = 1
        return result


if __name__ == "__main__":
    train_data = read_train_file("BDCI2017-minglue/1-train/train.txt")
    LawsClassifier(train_data, 10, 10)