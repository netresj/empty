# -*- coding:utf-8 -*-

import numpy as np

# 可視化に使用
import matplotlib.pyplot as plt

# データの読み込み
from tensorflow.keras.datasets.mnist import load_data


(X_train, y_train), (X_test, y_test) = load_data()


def data_visualize(image, label, predict=None):
    plt.figure(figsize=(5, 5))
    title = "label " + str(label)
    if predict != None:
        title += ", predict " + str(predict)
    plt.title(title)
    plt.imshow(image)
    plt.gray()
    plt.savefig("/kqi/output/figure/fig.png")


for i in range(3):
    data_visualize(X_train[i], y_train[i])
