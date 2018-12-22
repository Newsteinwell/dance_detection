import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
class PrepareData():
    def __init__(self, max=1.0, min=-1.0):
        self.dance_xiuse_path = '/Users/yc1/Documents/Lu/dance_detection/data/dance/dance_xiuse/'
        self.dance_mifeng_path = '/Users/yc1/Documents/Lu/dance_detection/data/dance/dance_mifeng/'
        self.sit_xiuse_path = '/Users/yc1/Documents/Lu/dance_detection/data/sit/xiuse/'
        self.sit_mifeng_path = '/Users/yc1/Documents/Lu/dance_detection/data/sit/mifeng/'
        self.dance_1_path = self.dance_xiuse_path + '1120/7027115/'
        self.dance_2_path = self.dance_xiuse_path + '1121/7018633/'
        self.dance_3_path = self.dance_xiuse_path + '1122/7045971/'
        self.dance_4_path = self.dance_xiuse_path + '1123/7034720/'
        self.dance_5_path = self.dance_xiuse_path + '1124/7018633/'
        self.dance_6_path = self.dance_xiuse_path + '1125/7047338/'
        self.dance_7_path = self.dance_mifeng_path + '108212/'
        self.dance_8_path = self.dance_mifeng_path + '108858/'
        self.dance_9_path = self.dance_mifeng_path + '111120/'
        self.dance_10_path = self.dance_mifeng_path + '114428/'
        self.dance_11_path = self.dance_mifeng_path + '116099/'
        self.dance_12_path = self.dance_mifeng_path + '117575/'

        self.sit_1_path = self.sit_xiuse_path + '7005743/'
        self.sit_2_path = self.sit_xiuse_path + '7007255/'
        self.sit_3_path = self.sit_xiuse_path + '7007489/'
        self.sit_4_path = self.sit_xiuse_path + '7009424/'
        self.sit_5_path = self.sit_xiuse_path + '7024175/'
        self.sit_6_path = self.sit_xiuse_path + '7026182/'
        self.sit_7_path = self.sit_mifeng_path + '100178/'
        self.sit_8_path = self.sit_mifeng_path + '100217/'
        self.sit_9_path = self.sit_mifeng_path + '100793/'
        self.sit_10_path = self.sit_mifeng_path + '101759/'
        self.sit_11_path = self.sit_mifeng_path + '102074/'
        self.sit_12_path = self.sit_mifeng_path + '102074/'

    def load_resize_norm_data(self):
        """
        1. Load images and resize. The size before are (320, 240) and (270, 480).
        2. Normalize the image. The use max min normalization, the default: max=1, min=-1
        :return: dance_pic, dance pictures data with resize and normalize.
                 sit_pic, sit pictures data with resize and normalize.
        """
        dance_pic = []
        for i in range(8):  # set to 8 to limit the data size, it can be enlarge \
                            # according to the compute ability
            print (i)
            file_path = eval('self.dance_{}_path'.format(i+1))
            for j, pic_path in enumerate(os.listdir(file_path)):
                image = Image.open(file_path + pic_path)
                resize_image = np.array(image.resize((270, 240)))
                normalized_image = self.normalize_pic(resize_image)
                dance_pic.append(normalized_image)

        sit_pic = []
        for i in range(12):
            print (i)
            file_path = eval('self.sit_{}_path'.format(i+1))
            for j, pic_path in enumerate(os.listdir(file_path)):
                image = Image.open(file_path + pic_path)
                resize_image = np.array(image.resize((270, 240)))
                normalized_image = self.normalize_pic(resize_image)
                sit_pic.append(normalized_image)
        dance_pic = np.array(dance_pic).astype(np.float32)
        sit_pic = np.array(sit_pic).astype(np.float32)
        return (dance_pic, sit_pic)

    def normalize_pic(self, arr, max_=1.0, min_=-1.0):
        """
        :param arr: np.array,  the array to normalize
        :param max_: float, the maximum value when implement max min normalize
        :param min_: float, the minimum value when implement max min normalize
        :return:
        """
        arr = arr.astype('float')
        for i in range(3):
            minval = arr[..., i].min()
            maxval = arr[..., i].max()
            if minval != maxval:
                arr[..., i] -= minval
                arr[..., i] *= (max_ - min_) / (maxval - minval)
                arr[..., i] +=  min_
        return arr

    def get_data(self):
        """
        To get the data after preparation.
        :return: dance, the dance data after preparation.
                 sit , the sit data after preparation.
        """
        dance_data, sit_data = self.load_resize_norm_data()
        dance_ = dance_data[np.random.choice(dance_data.shape[0], sit_data.shape[0], replace=True)]
        index = np.arange(sit_data.shape[0])
        np.random.shuffle(index)
        dance = dance_[index]
        sit = sit_data[index]
        return (dance, sit)

"""
example: 
prep_data = PrepareData()
dance_data, sit_data = prep_data.get_data()
print (dance_data)
print (sit_data)
"""

