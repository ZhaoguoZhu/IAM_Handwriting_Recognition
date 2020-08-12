from Characters_Dictionary import Dict
from Filename_To_Words_Dictionary import DictY
from os import listdir
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from numpy import save, asarray
from random import random


def prepare_data():
    D1, D2 = Dict(), DictY()
    directory = "words/"
    TrainX, TrainY, TestX, TestY= list(), list(), list(), list()
    for sub1 in listdir(directory):
        for sub2 in listdir(directory + sub1 + '/'):
            for files in listdir(directory + sub1 + '/' + sub2 + '/'):
                path = directory + sub1 + '/' + sub2 + '/' + files
                filename = files[:-4]
                photo = load_img(path,color_mode="grayscale", target_size=(32, 128))
                photo = img_to_array(photo)
                word_corresponding = D2.get(filename)
                label = [0] *32
                for i in range(len(word_corresponding)):
                    label[i] = D1.get(word_corresponding[i])
                label = asarray(label)
                if random()<0.5:
                    TestX.append(photo)
                    TestY.append(label)
                else:
                    TrainX.append(photo)
                    TrainY.append(label)

    TrainSample = asarray(TrainX)
    #print(type(TrainSample))
    #print(TrainSample)
    TrainLabel = asarray(TrainY)
    #print(type(TrainLabel))
    #print(TrainLabel)
    TestSample = asarray(TestX)
    TestLabel = asarray(TestY)

    '''
    print(TrainSample.shape, TrainLabel.shape)
    print(TestSample.shape, TestLabel.shape)    
    save('32x128_GrayscaleTrS.npy', TrainSample)
    save('32x128_GrayscaleTrL.npy', TrainLabel)
    save('32x128_GrayscaleTeS.npy', TestSample)
    save('32x128_GrayscaleTeL.npy', TestLabel)
    '''

    return TrainSample, TrainLabel, TestSample, TestLabel
