import os

import numpy as np
from skimage import io
import random as rnd


class ImagesDataset:
    def __init__(self, path, train_percent, shuffle_rounds, reversed_colors = False, dbg=False):
        self.x_train = list()
        self.y_train = list()
        self.x_test = list()
        self.y_test = list()
        self.path = path
        self.tp = train_percent
        self.shuffle_rounds = shuffle_rounds
        self.__create_dataset(reversed_colors, dbg)

    def __create_dataset(self, reversed_colors: bool, dbg: bool):
        folders = [e for e in os.listdir(self.path)]
        files = [list() for p in range(len(folders))]
        for fold_id in range(len(folders)):
            for fl in os.listdir(self.path + folders[fold_id]):
                files[fold_id].append(fl)
        classes = list(np.linspace(0, len(folders) - 1, len(folders)))
        x_raw = [list() for par in range(len(classes))]
        y_raw = [list() for par in range(len(classes))]
        for class_id in range(len(classes)):
            for sample_id in range(len(files[class_id])):
                x_raw[class_id].append(np.asarray(io.imread(self.path + folders[class_id] + "\\"
                                                            + files[class_id][sample_id])))
                x_raw[class_id][sample_id] = x_raw[class_id][sample_id][:, :, 0:3]
            y_raw[class_id].append(class_id)

        if dbg:
            print("Pre-transform data:")
            print("x shape:", np.shape(x_raw))
            print("y shape:", np.shape(y_raw))
            print("y:", y_raw)

        x_tmp = list(np.concatenate(x_raw))
        y_tmp = list()
        for class_id in range(len(classes)):
            for sample_id in range(len(x_raw[class_id])):
                y_tmp.append(y_raw[class_id])
        y_raw = [elem for elem in y_tmp]
        x_raw = [elem for elem in x_tmp]

        if dbg:
            print("Post-transform data:")
            print("x shape:", np.shape(x_raw))
            print("y shape:", np.shape(y_raw))
            print("y:", y_raw)

        self.x_train, self.y_train, self.x_test, self.y_test = self.__split(x_raw, y_raw)

        self.x_train, self.y_train = self.__shuffle(self.x_train, self.y_train)
        self.x_test, self.y_test = self.__shuffle(self.x_test, self.y_test)

        self.x_train = list(np.array(self.x_train) / 255.0)
        self.x_test = list(np.array(self.x_test) / 255.0)

        self.x_train = self.__reverse_images(self.x_train, reversed_colors)
        self.x_test = self.__reverse_images(self.x_test, reversed_colors)

        if dbg:
            print("Final dataset")
            print("x_train shape: ", np.shape(self.x_train))
            print("y_train shape: ", np.shape(self.y_train))
            print("x_test shape: ", np.shape(self.x_test))
            print("y_test shape: ", np.shape(self.y_test))
            print("y_train: ", self.y_train)
            print("y_test: ", self.y_test)

    def __reverse_images(self, images, reversed_colors):
        for image in images:
            for row in image:
                for pixel in row:
                    for c_id in range(len(pixel)):
                        if reversed_colors:
                            pixel[c_id] = 1 if pixel[c_id] < 0.5 else 0
                        else:
                            pixel[c_id] = 1 if pixel[c_id] > 0.5 else 0

        return images

    def __split(self, x, y):
        h_len = int(len(y) / 2)
        x_train = x[0: int(h_len * self.tp)]
        x_train.extend(list(x[h_len: int(h_len + h_len * self.tp)]))
        y_train = list(y[0: int(h_len * self.tp)])
        y_train.extend(list(y[h_len: int(h_len + h_len * self.tp)]))
        x_test = x[int(h_len * self.tp): h_len]
        x_test.extend(x[int(h_len + h_len * self.tp): h_len * 2])
        y_test = y[int(h_len * self.tp): h_len]
        y_test.extend(y[int(h_len + h_len * self.tp): h_len * 2])

        return x_train, y_train, x_test, y_test

    def __shuffle(self, x, y):
        for sample_id in range(self.shuffle_rounds):
            first_id = rnd.randint(0, len(x) - 1)
            second_id = rnd.randint(0, len(x) - 1)

            tmp = x[first_id]
            x[first_id] = x[second_id]
            x[second_id] = tmp

            tmp = y[first_id]
            y[first_id] = y[second_id]
            y[second_id] = tmp

        return x, y

