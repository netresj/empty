# -*- coding:utf-8 -*-

import numpy as np

# 可視化に使用
import matplotlib.pyplot as plt
import seaborn as sns

# 学習に使う keras モジュール
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# 精度評価などに使用するライブラリ
from sklearn.metrics import accuracy_score, confusion_matrix

(X_train, y_train), (X_test, y_test) = load_data()

def data_visualize(image, label, file_name, predict=None):
    plt.figure(figsize=(5, 5))
    title = "label " + str(label)
    if predict != None:
        title += ", predict " + str(predict)
    plt.title(title)
    plt.imshow(image)
    plt.gray()
    plt.savefig("/kqi/output/figure/" + file_name + ".png")

for i in range(3):
    data_visualize(X_train[i], y_train[i], "input_data/fig" + str(i))

X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255

y_train = to_categorical(y_train)

def build_model():
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(200, activation="relu"))
    model.add(Dense(200, activation="relu"))
    model.add(Dense(10, activation="softmax"))
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy")
    return model

model = build_model()

model.fit(X_train, y_train)

y_predict = model.predict(X_test)

y_predict = np.argmax(y_predict, axis=1)

cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, cmap='Blues', annot=True, square=True)
plt.savefig("/kqi/output/figure/confution_matrix.png")
for i in range(cm.shape[0]):
    cm[i, i] = 0
sns.heatmap(cm, cmap='Blues', annot=True, square=True)
plt.savefig("/kqi/output/figure/confution_matrix_v2.png")
