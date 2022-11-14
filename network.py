import math
import random as rnd
import matplotlib.pyplot as plt

import numpy as np


class PANetwork:
    def __init__(self,
                 layers,
                 l_neurons,
                 out_layer_size,
                 image_size,
                 lambd=0.1,
                 dbg=False):
        self.MAX_SIZE = 2048
        if layers * l_neurons > self.MAX_SIZE:
            raise Exception("Too many items")
        self.lambd = lambd
        self.dbg = dbg
        self.image_size = image_size
        self.w_bounds = -1, 1
        self.__layers = layers
        self.__layer_neurons = l_neurons
        self.ol_size = out_layer_size
        self.weights = list()
        self.__initialize_weights()

    def train(self,
              x_train,
              y_train,
              x_test,
              y_test,
              learning_rate,
              ages=800):
        y_train = list(y_train)
        loss_func = list()
        ages_func = list()
        for age_id in range(ages):
            sample_id = rnd.randint(0, len(x_train) - 1)
            curr_x = x_train[sample_id]
            percents, outputs = self.__get_outputs(curr_x)
            y_predicted = -1
            y_true = y_train[sample_id][0]
            mx = max(percents)
            for per_id in range(len(percents)):
                if percents[per_id] == mx:
                    y_predicted = per_id
            ages_func.append(age_id + 1)
            hit_rate, loss = self.test(x_test, y_test)
            loss_func.append(loss)
            if hit_rate > 0.9:
                print("\n\n\n\nTraining ended\nTest split hit rate: ", str(hit_rate))
                break
            delta = y_true - y_predicted
            if y_predicted != y_train[sample_id]:
                for layer_id in range(len(outputs) - 1):
                    for neuron_id in range(len(outputs[layer_id])):
                        c_w = np.array(self.weights[layer_id][neuron_id])
                        out = outputs[layer_id + 1]
                        out = out + [1]
                        for w_id in range(len(c_w)):
                            c_w[w_id] = c_w[w_id] - learning_rate * 2 * delta * self.__activation_dev(c_w, out)\
                                        * out[w_id]
                        self.weights[layer_id][neuron_id] = list(c_w)
            if self.dbg and (age_id + 1) % 1 == 0:
                print("after age ", str(age_id + 1), " weights are:\n",
                      str(self.weights[:len(self.weights) - 1]), "\n",
                      "hit rate: ", hit_rate,
                      "loss: ", loss)
        plt.plot(ages_func, loss_func)
        plt.show()

    def test(self, x_test, y_test):
        hits = 0
        loss = 0
        for rule_id in range(len(x_test)):
            percents, outputs = self.__get_outputs(x_test[rule_id])
            y_true = y_test[rule_id][0]
            y_predicted = -1
            mx = max(percents)
            for per_id in range(len(percents)):
                if percents[per_id] == mx:
                    y_predicted = per_id
            if y_predicted == y_true:
                hits = hits + 1
            loss += (y_predicted - y_true) ** 2
            loss = math.sqrt(loss)
        hit_rate = float(hits) / len(x_test)
        return hit_rate, loss

    def __get_outputs(self, rule):
        """if dbg:
            print("start weights: ")
            for layer in self.weights:
                print(layer, "\n")"""
        outputs = list()
        for i in range(len(self.weights)):
            layer = list()
            for j in range(len(self.weights[i])):
                layer.append(-1)
            outputs.append(layer)
        """if self.dbg:
            print("Then")
            print("weights shape: ", np.shape(self.weights))
            print("outputs shape: ", np.shape(outputs))"""
        for layer_id in reversed(range(self.__layers + 1)):
            for neuron_id in range(len(self.weights[layer_id])):
                weights = self.weights[layer_id][neuron_id]
                if layer_id == self.__layers:
                    signals = list()
                    for line in rule:
                        for column in line:
                            for rgb_comp in column:
                                signals.append(rgb_comp)
                    signals.append(1)
                else:
                    signals = list(outputs[layer_id + 1])
                    signals.append(1)
                outputs[layer_id][neuron_id] = self.__is_activated(weights, signals, layer_id)
        ''' if self.dbg:
            print("Now")
            print("weights shape: ", np.shape(self.weights))
            print("outputs shape: ", np.shape(outputs))'''

        return outputs[0], outputs

    '''def __is_activated(self, weights, outputs, layer_id):
        x = np.dot(weights, outputs)
        if layer_id != 0:
            return 1 if 1 / (1 + math.exp(-x)) >= 0.5 else 0
        elif layer_id != -999:
            return 1 / (1 + math.exp(-x))
        else:
            return -1

    def __activation_dev(self, weights, outputs):
        x = np.dot(weights, outputs)

        return self.__is_activated(weights, outputs, 0) * (1 - self.__is_activated(weights, outputs, 0))'''

    def __is_activated(self, weights, outputs, layer_id):
        x = np.dot(weights, outputs)
        if layer_id != 0:
            return 1 if (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x)) >= 0 else 0
        elif layer_id != -999:
            return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
        else:
            return -1

    def __activation_dev(self, weights, outputs):
        x = np.dot(weights, outputs)

        return 1 - (self.__is_activated(weights, outputs, 0)) ** 2

    def __initialize_weights(self):
        layer = list()
        for neu_id in range(self.ol_size):
            neuron = list()
            for w_id in range(self.__layer_neurons + 1):
                neuron.append(rnd.randint(self.w_bounds[0], self.w_bounds[1]) / 2)
            layer.append(neuron)

        self.weights.append(layer)

        for layer_id in range(self.__layers - 1):
            layer = list()
            for neu_id in range(self.__layer_neurons):
                neuron = list()
                for w_id in range(self.__layer_neurons + 1):
                    neuron.append(rnd.randint(self.w_bounds[0], self.w_bounds[1]) / 2)
                layer.append(neuron)

            self.weights.append(layer)

        layer = list()
        for neu_id in range(self.__layer_neurons):
            neuron = list()
            for w_id in range((self.image_size ** 2) * 3 + 1):
                neuron.append(rnd.randint(self.w_bounds[0], self.w_bounds[1]) / 2)
            layer.append(neuron)

        self.weights.append(layer)

