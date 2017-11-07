#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import os.path

import tensorflow as tf
import numpy as np

DICT_PATH = "word_dict.txt"
LAW_NUM = 453
HIDDEN_LAYER_NUM = 64
TEMP_PATH = "tmp"
SOURCE_FILE_NAME = "source_data"

if not tf.gfile.IsDirectory(TEMP_PATH):
    tf.gfile.MakeDirs(TEMP_PATH)


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


class LawsClassifier:
    """
    对样本所属法律条文的分类器，根据法律条文里的关键字，统计该案例中这些关键字的出现频率，并以频率向量作为输入，所属条文
    作为输出（0，1表示），建立单层神经网络模型进行训练并获取结果模型
    """
    def __init__(self, source_data, testing_percentage, validation_percentage):
        """
        实例化该类即开始转换
        :param source_data: 经过read_train_file过程获得的源数据列表
        :param testing_percentage: 测试集百分比
        :param validation_percentage: 验证集百分比
        """
        # TODO: 将训练过程标准化，从类构造函数中移出
        # TODO: 添加验证集调整参数功能
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
        # TODO: 优化损失函数，当前计算速度过慢
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(laws_placeholder - prediction)))
        train_step = tf.train.GradientDescentOptimizer(0.000001).minimize(loss)

        init = tf.global_variables_initializer()
        print("Starting session...")
        with tf.Session() as sess:
            sess.run(init)
            # TODO: 添加保存记录点语句等
            for i in range(1000):
                sess.run(train_step, feed_dict={text_placeholder: self.training_text_list,
                                                laws_placeholder: self.training_laws_list})
                print("Training loss: {}".format(sess.run(loss, feed_dict={text_placeholder: self.training_text_list,
                                                                           laws_placeholder: self.training_laws_list})))
                print("Testing loss: {}".format(sess.run(loss, feed_dict={text_placeholder: self.testing_text_list,
                                                                          laws_placeholder: self.testing_laws_list})))

        # TODO: 添加保存模型功能
        # TODO: 添加获取测试集样本准确率功能

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
    def open_dict(dict_path):
        """
        读取词典文件，因为该文件由jieba分词工具直接对‘法律条文.txt’进行分词获得，需要移除数字等无意义关键词
        :param dict_path: 词典文件路径
        :return: 关键词的列表
        """
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
    def processed_law(law_list):
        """
        获得该案例相应法律条文号组成的列表
        :param law_list: 法律条文号列表
        :return: 一个‘总法律条文数目’长度维度的向量，如果出现该法律条文，该位置置1，否则置0
        """
        result = [0 for _ in range(453)]
        for law in law_list:
            result[law] = 1
        return result

# TODO: 增加罚金金额分类模型类


if __name__ == "__main__":
    # 根据实际情况修改文件名路径
    train_data = read_train_file("BDCI2017-minglue/1-train/train.txt")
    LawsClassifier(train_data, 10, 10)
