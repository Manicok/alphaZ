#!/usr/bin/env python
# coding: utf-8

# In[1]:

import tensorflow as tf
import keras
from keras import layers
from keras import backend as K


# In[2]:


class ResBlock(keras.Model):
    """残差块"""
    def __init__(self, name, filters, kernel_size, data_format, l2_penalty):
        super().__init__(name=name)

        self.conv2d_1 = layers.Conv2D(filters, kernel_size, padding="same", data_format=data_format, kernel_regularizer=keras.regularizers.l2(l2_penalty))
        self.batch_1 = layers.BatchNormalization()

        self.conv2d_2 = layers.Conv2D(filters, kernel_size, padding="same", data_format=data_format, kernel_regularizer=keras.regularizers.l2(l2_penalty))
        self.batch_2 = layers.BatchNormalization()

    def call(self, Input, training=False):
        x = self.conv2d_1(Input)
        x = self.batch_1(x, training=training)
        x = layers.ReLU()(x)
        x = self.conv2d_2(x)
        x = self.batch_2(x, training=training)
        x += Input
        return layers.ReLU()(x)


# In[32]:


class PolicyValueNet:
    """策略价值网络"""
    def __init__(self, board_length=19, data_format="channels_first", l2_penalty=1e-4, initial_model=None):
        if initial_model:
            self.model = keras.models.load_model(initial_model)
            return
        Input = layers.Input((17, board_length, board_length))
        #  卷积层
        x = keras.Sequential([
            layers.Conv2D(256, (3, 3), padding="same", data_format=data_format, kernel_regularizer=keras.regularizers.l2(l2_penalty)),
            layers.BatchNormalization(),
            layers.ReLU()
        ])(Input)
        #  残差塔
        x = keras.Sequential([
            ResBlock(f"ResBlock{i}", 256, (3, 3), data_format, l2_penalty) for i in range(39)
        ])(x)
        #  策略头
        policy = keras.Sequential([
            layers.Conv2D(2, (1, 1), data_format=data_format, kernel_regularizer=keras.regularizers.l2(l2_penalty)),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Flatten(),
            layers.Dense(board_length**2 + 1, activation="softmax", kernel_regularizer=keras.regularizers.l2(l2_penalty))
        ])(x)
        #  价值头
        value = keras.Sequential([
            layers.Conv2D(1, (1, 1), data_format=data_format, kernel_regularizer=keras.regularizers.l2(l2_penalty)),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Flatten(),
            layers.Dense(256, kernel_regularizer=keras.regularizers.l2(l2_penalty)),
            layers.ReLU(),
            layers.Dense(1, activation="tanh", kernel_regularizer=keras.regularizers.l2(l2_penalty))
        ])(x)

        self.model = keras.Model(inputs=Input,
                                 outputs=[policy, value])
        self.model.compile(optimizer=tf.keras.optimizers.SGD(momentum=.9),
                           loss=['mean_squared_error', 'mean_squared_error'])

    def PolicyValueFunction(self, board):
        action_space = list(board.current_space)
        data = board.get_data()
        policy, value = self.model.predict(data[None, :], verbose=0)
        action_prob_pairs = list(zip(action_space, policy[0][action_space]))
        return action_prob_pairs, value

    def fit(self, states, search_probabilities, winner, learning_rate, batch_size):
        self.model.optimizer.learning_rate = learning_rate
        self.model.fit(states, [search_probabilities, winner], batch_size=batch_size)

    def save_model(self, name):
        name = name + '.keras'
        self.model.save(name)

