import json
import random

import cv2  # working with, mainly resizing, images
import numpy as np  # dealing with arrays
import os  # dealing with directories
from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

path = "./Dataset/data"
IMG_SIZE = 50
LR = 1e-3
MODEL_NAME = 'healthyvsunhealthy-{}-{}.model'.format(LR, '2conv-basic')
listdir = os.listdir(path)
shuffle(listdir)
temp = open("data.json")
temp1 = json.load((temp))

def label_img(img):
    global temp1
    word_label = temp1[img]

    if word_label == 'A':
        return [1, 0, 0, 0, 0]

    elif word_label == 'B':
        return [0, 1, 0, 0, 0]
    elif word_label == 'C':
        return [0, 0, 1, 0, 0]
    elif word_label == 'D':
        return [0, 0, 0, 1, 0]
    elif word_label == "g":
        return [0, 0, 0, 0, 1]


def create_train_data():
    global listdir,path
    i = 0
    # training_data = []
    training_x, training_y = [], []
    for img in tqdm(listdir[:-100]):
        label = label_img(img)
        temp_path = os.path.join(path, img)
        img = cv2.imread(temp_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        # training_data.append([np.array(img), np.array(label)])
        training_x.append(np.array(img))
        training_y.append(np.array(label))
    # shuffle(training_data)
    # np.save('train_data.npy', training_data)
    return training_x, training_y


def process_test_data():
    global listdir,path
    testing_x = []
    testing_y = []
    for img in tqdm(listdir[-100:]):
        label = label_img(img)
        temp_path = os.path.join(path, img)
        img = cv2.imread(temp_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        testing_x.append(np.array(img))
        testing_y.append(np.array(label))
    return testing_x,testing_y


# train_data = create_train_data()
temp_x, temp_y = create_train_data()
# If you have already created the
# dataset:
# train_data = np.load('train_data.npy')


import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf

# tf.reset_default_graph()

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

convnet = conv_2d(convnet, 32, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 64, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 128, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 32, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 64, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.5)

convnet = fully_connected(convnet, 5, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

# network = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3])
# network = conv_2d(network, 32, 3, activation='relu')
# network = max_pool_2d(network, 2)
# network = conv_2d(network, 64, 3, activation='relu')
# network = max_pool_2d(network, 2)
# network = conv_2d(network, 128, 3, activation='relu')
# network = max_pool_2d(network, 2)
# # network = fully_connected(network, 32, activation='relu')
# # network = dropout(network, 0.5)
# network = fully_connected(network, 1024, activation='relu')
# network = dropout(network, 0.5)
# network = fully_connected(network, 5, activation='softmax')
# network = regression(network, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)

# model = tflearn.DNN(network, tensorboard_verbose = 0)

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')

else:
    # train = train_data[:-500]
    # test = train_data[-500:]

    X, Y = temp_x[:len(temp_x)-200], temp_y[:len(temp_y)-200]
    test_x, test_y = temp_x[-200:], temp_y[-200:]

    # X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    # Y = [i[1] for i in train]
    #
    # test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    # test_y = [i[1] for i in test]

    model.fit({'input': X}, {'targets': Y}, n_epoch=8, validation_set=({'input': test_x}, {'targets': test_y}),
            snapshot_step=40, show_metric=True, run_id=MODEL_NAME)

testing_x, testing_y = process_test_data()
# testing_x = np.array(testing_x).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
model.save(MODEL_NAME)

test_no = random.randint(0, len(testing_x))
prediction = model.predict(testing_x)

def get_label(l):
    return ["Aanthracnose","Bacterial blight","Cercospora Fruit Spot","Fruit Rot","Healthy Fruit"][np.argmax(np.array(l))]

def get_one_hot(l):
    index = np.argmax(l)
    one_hot = [0, 0, 0, 0, 0]
    one_hot[index] = 1
    return one_hot

# acc_x = [get_one_hot(i) for i in prediction]

for i in range(len(prediction)):
    print(get_label(prediction[i]), get_label(testing_y[i]))

pred = model.predict(testing_x)
pred_x = [get_one_hot(i) for i in pred]

print(f"Testing Accuracy : {accuracy_score(testing_y,pred_x)*100}")
# img = "d15.jpg"
val_x = []
val_y = []
for img in tqdm(os.listdir("testing")):
    temp_path = os.path.join("testing", img)
    # print(img, temp_path)
    label = label_img(img)
    img = cv2.imread(temp_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    val_x.append(np.array(img))
    val_y.append(np.array(label))

prediction = model.predict(val_x)

acc_x = [get_one_hot(i) for i in prediction]

precision = precision_score(val_y, acc_x, average=None)
recall = recall_score(val_y, acc_x, average=None)
f1 = f1_score(val_y, acc_x, average=None)

tn = []
fp = []

conf_pred, conf_true = np.array([get_label(i) for i in acc_x]), np.array([get_label(i) for i in val_y])

# print( confusion_matrix(conf_true == "A", conf_pred=="A"))
for i in ["Aanthracnose","Bacterial blight","Cercospora Fruit Spot","Fruit Rot","Healthy Fruit"]:
    tn_i, fp_i = confusion_matrix(conf_true == i, conf_pred == i)[0]
    tn.append(tn_i)
    fp.append(fp_i)
specificity = []

for i in range(5):
    specificity_i = tn[i] / (tn[i] + fp[i])
    specificity.append(specificity_i)

# for i in range(len(val_y)):
    # print(f"acc: {acc_x[i]} \nPrediction: {prediction[i]}\nVal_y: {val_y[i]}")
print(f"validation Accuracy : {accuracy_score(acc_x, val_y)*100} \n Precision : {precision} \n Recall(Sensitivity) : {recall} \n F1 score : {f1} \n Specificity : {specificity}")
# print(f"accuracy: {int(accuracy_score(acc_x, val_y)*100)} %" )