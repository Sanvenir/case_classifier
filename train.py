#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import os.path
import json
from collections import OrderedDict

import tensorflow as tf
import numpy as np

# 字典文件路径
DICT_PATH = "word_dict.txt"
# 法律条文总数
LAW_NUM = 452
# 隐藏层节点数
HIDDEN_LAYER_NUM = 64
# 训练用缓存文件存储位置
TEMP_PATH = "tmp"
# 测试用缓存文件存储位置
TEST_TEMP_PATH = "test"
# 模型存储位置
SAVE_PATH = "sav"
# 单步训练数
TRAINING_NUM = 1000
# 获取测试精确度的训练步数
TEST_PATH_NUM = 50
# 法律条文输出置信度阈值
LAW_THRESHOLD = 0.0001
# 罚金范围
PENALTY_NUM = 8


word_num = None

if not tf.gfile.IsDirectory(TEMP_PATH):
    tf.gfile.MakeDirs(TEMP_PATH)

if not tf.gfile.IsDirectory(TEST_TEMP_PATH):
    tf.gfile.MakeDirs(TEST_TEMP_PATH)


class Classifier:
    """
    分类器父类
    """
    def __init__(self):
        self.word_dict = LawsClassifier.open_dict()
        self.training_text_list = []
        self.training_laws_list = []
        self.testing_text_list = []
        self.testing_laws_list = []
        self.validation_text_list = []
        self.validation_laws_list = []
        self._prediction = None
        self._loss = None
        self._optimizer = None

    @property
    def prediction(self):
        return self._prediction

    @property
    def loss(self):
        return self._loss

    @property
    def optimizer(self):
        return self._optimizer

    @staticmethod
    def get_split_data_pos(source_size, data_size, prev_step):
        return (prev_step * data_size) % source_size - source_size

    @staticmethod
    def add_layer(inputs, in_size, out_size, activation_function=None):
        """
        添加神经网络层
        :param inputs: 隐藏层输入向量
        :param in_size: 输入节点个数
        :param out_size: 输出节点个数
        :param activation_function: 激活函数
        :return:
        """
        weights = tf.Variable(tf.random_normal([in_size, out_size]))
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        wx_plus_b = tf.matmul(inputs, weights) + biases
        if activation_function is None:
            outputs = wx_plus_b
        else:
            outputs = activation_function(wx_plus_b)
        return outputs

    @staticmethod
    def open_dict():
        """
        读取词典文件，因为该文件由jieba分词工具直接对‘法律条文.txt’进行分词获得，需要移除数字等无意义关键词
        :return: 关键词的列表
        """
        dict_path = DICT_PATH
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
        global word_num
        word_num = len(result)
        return result

    @staticmethod
    def processed_string(words_dict, string, file_name=None):
        """
        如果预处理文件存在，则读入文件并返回文件内容；否则，将案例的文字描述表示成各关键词出现频率的向量，并保存至文件中
        :param words_dict: 关键词列表
        :param string: 案例文字描述
        :param file_name: 结果保存文件名
        :return: 关键词出现频率列表，index与关键词列表中一致，value为该关键词出现的次数，会在神经网络模型中归一化处理
        """
        if tf.gfile.Exists(file_name):
            return np.fromfile(file_name, dtype=np.float32)
        result = np.array([string.count(word) for word in words_dict], dtype=np.float32)
        if file_name is not None:
            result.tofile(file_name)
        return result

    @staticmethod
    def read_train_file(file_name):
        """
        读取训练文件，并将数据存入list列表中
        :param file_name: 训练文件名(string)
        :return: 数据list，每一个元素为[样本id(int)，样本文字(string)，样本罚金金额标签(int)，样本涉及法律条文（int list)
        """
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

    @staticmethod
    def read_test_file(file_name):
        """
        读取测试文件
        :param file_name:
        :return:
        """
        if not tf.gfile.Exists(file_name):
            raise FileNotFoundError("Train file '{}' not found".format(file_name))

        result = dict()
        with tf.gfile.Open(file_name) as test_file:
            while True:
                line = test_file.readline()
                if not line:
                    break
                elements = line.split('\t')
                result[int(elements[0])] = elements[1]
        return result


class PenaltyClassifier(Classifier):
    """
    对样本惩罚金额的分类器
    """
    def __init__(self):
        """
        初始化各tensor
        """
        super().__init__()

        # 定义输入、输出标准
        self.text_placeholder = tf.placeholder(tf.float32, shape=[None, word_num])
        self.penalty_placeholder = tf.placeholder(tf.float32, shape=[None, PENALTY_NUM])

        self._prediction = self.prediction
        self._loss = self.loss
        self._optimizer = self.optimizer

        self.predict = tf.argmax(self.prediction, 1)
        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(self.predict, tf.argmax(self.penalty_placeholder, 1)), tf.float32))

        self.init = tf.global_variables_initializer()

    @property
    def prediction(self):
        """
        :return: 模型预测值输出（罚金各标签的置信度）
        """
        if self._prediction is not None:
            return self._prediction
        # 神经网络模型
        text_normalize = tf.nn.softmax(self.text_placeholder)
        l1 = LawsClassifier.add_layer(text_normalize, word_num, HIDDEN_LAYER_NUM,
                                      activation_function=tf.nn.relu)
        return LawsClassifier.add_layer(l1, HIDDEN_LAYER_NUM, PENALTY_NUM, activation_function=None)

    @property
    def loss(self):
        """
        :return: 损失函数
        """
        if self._loss is not None:
            return self._loss
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            logits=self.prediction, labels=self.penalty_placeholder)
        return tf.reduce_mean(cross_entropy)

    @property
    def optimizer(self):
        """
        :return: 最小化损失函数方法
        """
        if self._optimizer is not None:
            return self._optimizer
        return tf.train.GradientDescentOptimizer(0.01).minimize(self.loss)

    def train(self, source_data, testing_percentage, validation_percentage):
        """
        训练模型
        :param source_data:
        :param testing_percentage:
        :param validation_percentage:
        :return:
        """
        self.load_data(source_data, testing_percentage, validation_percentage)
        print("Complete.")
        saver = tf.train.Saver()
        print("Starting session...")
        with tf.Session() as sess:
            sess.run(self.init)
            for i in range(1000):
                pos = self.get_split_data_pos(len(self.training_text_list), TRAINING_NUM, i)
                if pos + TRAINING_NUM >= 0:
                    _, training_loss = sess.run([self.optimizer, self.loss], feed_dict={
                        self.text_placeholder:
                            self.training_text_list[pos:] + self.training_text_list[:pos + TRAINING_NUM],
                        self.penalty_placeholder:
                            self.training_laws_list[pos:] + self.training_laws_list[:pos + TRAINING_NUM]})
                else:
                    _, training_loss = sess.run([self.optimizer, self.loss], feed_dict={
                        self.text_placeholder: self.training_text_list[pos: pos + TRAINING_NUM],
                        self.penalty_placeholder: self.training_laws_list[pos: pos + TRAINING_NUM]})
                if i % TEST_PATH_NUM == 0:
                    testing_accuracy = sess.run(self.accuracy, feed_dict={
                        self.text_placeholder: self.testing_text_list,
                        self.penalty_placeholder: self.testing_laws_list})

                    print("Step {} Training loss: {}".format(i, training_loss))
                    print("Step {} Testing accuracy: {}".format(i, testing_accuracy))
            tf.gfile.MakeDirs(SAVE_PATH)
            saver.save(sess, os.path.join(SAVE_PATH, "penalty_model.tf"))
            print("Model saved")

    def classifier(self, test_data, result):
        """
        对测试数据进行分类并获得分类结果
        :param test_data:
        :param result: 结果字典
        :return:
        """
        assert isinstance(test_data, dict)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(self.init)
            load_path = saver.restore(sess, os.path.join(SAVE_PATH, "penalty_model.tf"))
            print("Load model from {}".format(load_path))
            count = 0
            for identity in test_data:
                text_vector = self.processed_string(self.word_dict, test_data[identity],
                                                    file_name=os.path.join(TEST_TEMP_PATH, str(identity)))
                penalty_predict = sess.run(
                    self.predict, feed_dict={self.text_placeholder: text_vector.reshape(1, len(self.word_dict))})
                result[identity]["penalty"] = int(penalty_predict[0] + 1)
                count += 1
                if not count % 1000:
                    print("Working on {}(id: {}), penalty predict is {}".format(count, identity, penalty_predict[0]))

    def load_data(self, source_data, testing_percentage, validation_percentage):
        """
        加载数据
        :param source_data: 源数据数组
        :param testing_percentage: 测试集百分比
        :param validation_percentage: 验证集百分比
        :return:
        """
        print("Preparing source data...")
        for i, elements in enumerate(source_data):
            if not i % 1000:
                print("{} / {} complete.".format(i, len(source_data)))
            checker = random.randrange(100)
            text_list = self.processed_string(self.word_dict, elements[1],
                                              file_name=os.path.join(TEMP_PATH, "{}.tmp".format(elements[0])))
            laws_list = self.processed_penalty(elements[2])
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
    def processed_penalty(penalty):
        """
        向量化罚金类别
        :param penalty:
        :return:
        """
        result = np.zeros(PENALTY_NUM)
        result[penalty - 1] = 1
        return result


class LawsClassifier(Classifier):
    """
    对样本所属法律条文的分类器，根据法律条文里的关键字，统计该案例中这些关键字的出现频率，并以频率向量作为输入，所属条文
    作为输出（0，1表示），建立单层神经网络模型进行训练并获取结果模型
    """
    def __init__(self):
        """
        初始化各tensor
        """
        super().__init__()
        # TODO: 添加验证集调整参数功能

        self.text_placeholder = tf.placeholder(tf.float32, shape=[None, word_num])
        self.laws_placeholder = tf.placeholder(tf.float32, shape=[None, LAW_NUM])

        self._prediction = self.prediction
        self._loss = self.loss
        self._optimizer = self.optimizer

        # 预测值（法律条文向量，True为对应该条法律条文）
        self.predict = tf.greater(tf.nn.softmax(self.prediction), LAW_THRESHOLD)

        # 预测值（法律条文序列）
        self.predict_result = tf.where(self.predict)

        # 准确度
        self.correct = tf.equal(self.laws_placeholder, 1.0)
        self.union = tf.logical_or(self.predict, self.correct)
        self.intersection = tf.logical_and(self.predict, self.correct)
        self.accuracy = tf.reduce_mean(tf.divide(tf.count_nonzero(self.intersection), tf.count_nonzero(self.union)))

        self.init = tf.global_variables_initializer()

    @property
    def prediction(self):
        """
        :return: 各法律条文相关的置信度
        """
        text_normalize = tf.nn.softmax(self.text_placeholder)
        if self._prediction is not None:
            return self._prediction
        l1 = self.add_layer(text_normalize, word_num, HIDDEN_LAYER_NUM,
                            activation_function=tf.nn.relu)
        return self.add_layer(l1, HIDDEN_LAYER_NUM, LAW_NUM, activation_function=None)

    @property
    def loss(self):
        """
        :return: 损失函数
        """
        if self._loss is not None:
            return self._loss
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction,
                                                                labels=self.laws_placeholder)
        return tf.reduce_mean(cross_entropy)

    @property
    def optimizer(self):
        """
        :return: 最小化损失函数方法
        """
        if self._optimizer is not None:
            return self._optimizer
        return tf.train.GradientDescentOptimizer(0.001).minimize(self.loss)

    def train(self, source_data, testing_percentage, validation_percentage):
        """
        对训练数据进行学习，获得并保存模型
        :param source_data:
        :param testing_percentage:
        :param validation_percentage:
        :return:
        """
        self.load_data(source_data, testing_percentage, validation_percentage)
        print("Complete.")
        saver = tf.train.Saver()
        print("Starting session...")
        with tf.Session() as sess:
            sess.run(self.init)
            for i in range(1000):
                pos = self.get_split_data_pos(len(self.training_text_list), TRAINING_NUM, i)
                if pos + TRAINING_NUM >= 0:
                    _, training_loss = sess.run([self.optimizer, self.loss], feed_dict={
                        self.text_placeholder:
                            self.training_text_list[pos:] + self.training_text_list[:pos + TRAINING_NUM],
                        self.laws_placeholder:
                            self.training_laws_list[pos:] + self.training_laws_list[:pos + TRAINING_NUM]})
                else:
                    _, training_loss = sess.run([self.optimizer, self.loss], feed_dict={
                        self.text_placeholder: self.training_text_list[pos: pos + TRAINING_NUM],
                        self.laws_placeholder: self.training_laws_list[pos: pos + TRAINING_NUM]})
                if i % TEST_PATH_NUM == 0:
                    testing_accuracy = sess.run(self.accuracy, feed_dict={
                        self.text_placeholder: self.testing_text_list,
                        self.laws_placeholder: self.testing_laws_list})

                    print("Step {} Training loss: {}".format(i, training_loss))
                    print("Step {} Testing accuracy: {}".format(i, testing_accuracy))
            tf.gfile.MakeDirs(SAVE_PATH)
            saver.save(sess, os.path.join(SAVE_PATH, "laws_model.tf"))
            print("Model saved")

    def classifier(self, test_data, result):
        """
        对测试数据进行法律条文分类，获得结果
        :param test_data:
        :param result: 结果字典
        :return:
        """
        assert isinstance(test_data, dict)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(self.init)
            load_path = saver.restore(sess, os.path.join(SAVE_PATH, "laws_model.tf"))
            print("Load model from {}".format(load_path))
            count = 0
            for identity in test_data:
                text_vector = self.processed_string(self.word_dict, test_data[identity],
                                                    file_name=os.path.join(TEST_TEMP_PATH, str(identity)))
                laws_predict = sess.run(
                    self.predict_result, feed_dict={self.text_placeholder: text_vector.reshape(1, word_num)})
                laws_predict = (np.transpose(laws_predict)[1] + 1).tolist()
                result[identity]["laws"] = laws_predict
                count += 1
                if count % 1000 == 0:
                    print("Working on {}(id: {}), law predict is {}".format(count, identity, laws_predict))

    def load_data(self, source_data, testing_percentage, validation_percentage):
        """
        加载数据
        :param source_data: 源数据数组
        :param testing_percentage: 测试集百分比
        :param validation_percentage: 验证集百分比
        :return:
        """
        print("Preparing source data...")
        for i, elements in enumerate(source_data):
            if not i % 1000:
                print("{} / {} complete.".format(i, len(source_data)))
            checker = random.randrange(100)
            text_list = self.processed_string(self.word_dict, elements[1],
                                              file_name=os.path.join(TEMP_PATH, "{}.tmp".format(elements[0])))
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
    def processed_law(law_list):
        """
        获得该案例相应法律条文号组成的列表
        :param law_list: 法律条文号列表
        :return: 一个‘总法律条文数目’长度维度的向量，如果出现该法律条文，该位置置1，否则置0
        """
        result = np.zeros(LAW_NUM, dtype=np.float32)
        for law in law_list:
            result[law - 1] = 1
        return result


def training_procedure(file_name):
    train_data = Classifier.read_train_file(file_name)
    PenaltyClassifier().train(train_data, testing_percentage=10, validation_percentage=0)
    LawsClassifier().train(train_data, testing_percentage=10, validation_percentage=0)


def testing_procedure(file_name, output_name):
    test_data = Classifier.read_test_file(file_name)
    test_result = OrderedDict()
    for identity in test_data:
        test_result[identity] = {"id": identity}
    PenaltyClassifier().classifier(test_data, test_result)
    LawsClassifier().classifier(test_data, test_result)
    with open(output_name, 'w') as f:
        for value in test_result.values():
            print(value)
            json.dump(value, f)
            f.write("\n")


if __name__ == "__main__":
    # 根据实际情况修改文件名路径
    # training_procedure("BDCI2017-minglue/1-train/train.txt")
    testing_procedure("BDCI2017-minglue/2-test/test.txt", "data.json")
