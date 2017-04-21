---
layout: post
title:  "predict xsin(x) using tensorflow"
date:   2017-04-21 03:20:19
categories: jekyll update
---

In this task, we want to use one `LSTM` cell to polyfit `xsin(x)` function.

code here:

```Python
'''
task : build minimum lstm prediction network
author : zyoohv@gmail.com
'''
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import copy


class GetConfig(object):
    """docstring for GetConfig"""
    learning_rate = 1e-2
    num_seq = 30
    hidden_size = 64

config = GetConfig()
origin_data = np.linspace(5, 55, 300)


def getTrainData():
    """
    format of inputs and outputs:
        inputs  : batch_size * num_seq * 1
        outputs : batch_size * num_seq
    """
    global config
    global origin_data

    inputs = []
    outputs = []
    for i in range(0, len(origin_data) - config.num_seq):
        thisInput = list(map(lambda x: x * np.sin(x), origin_data[i:i + config.num_seq]))
        inputs.append(np.array(thisInput).reshape(config.num_seq, 1))
        outputs.append(list(map(lambda x: x * np.sin(x), [origin_data[i + config.num_seq]])))
    return np.array(inputs), np.array(outputs)


def build_model():
    global config
    global input_data
    global target_data

    input_data = tf.placeholder(tf.float32, [None, config.num_seq, 1])
    target_data = tf.placeholder(tf.float32, [None, 1])

    cell = tf.contrib.rnn.BasicLSTMCell(config.hidden_size, state_is_tuple=True)
    # we don't add dropout layer here to train network more quickly
    # cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=1.0, output_keep_prob=1.0)
    val, state = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)

    # val : num_seq * batch_size * hidden_size
    val = tf.transpose(val, [1, 0, 2])
    las = tf.gather(val, config.num_seq - 1)

    W = tf.Variable(tf.truncated_normal([config.hidden_size, 1]))
    b = tf.Variable(tf.constant(0.1, shape=[1]))

    # batch_size * 1 * 1
    result = tf.matmul(las, W) + b
    return result


def main():
    global config
    global input_data
    global target_data
    global origin_data

    # plot the origin curve
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(origin_data, origin_data * np.sin(origin_data), 'k-')

    # define loss function and optimizer method here
    model = build_model()
    loss = tf.reduce_mean(tf.square(model - target_data))
    train_op = tf.train.AdamOptimizer(config.learning_rate).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(5000):
            inputs, outputs = getTrainData()
            result, loss_val, _ = sess.run([model, loss, train_op], feed_dict={input_data: inputs, target_data: outputs})
            if epoch % 100 == 0:
                print('epoch = {} loss = {:.6f}'.format(epoch / 100 + 1, loss_val))

        # test the lstm network
        step_len = float(origin_data[1] - origin_data[0])
        pre_x = []
        pre_y = []
        origin_data = origin_data[:-150]

        def f(data):
            return data * np.sin(data)

        for _ in range(200):
            pre_x.append(origin_data[-1] + step_len)
            now_input = f(origin_data[-config.num_seq:]).reshape(config.num_seq, 1)
            now_input = np.expand_dims(now_input, axis=0)
            result = sess.run([model], feed_dict={input_data: now_input})
            pre_y.append(result[0][0][0])
            origin_data = np.concatenate((origin_data, [pre_x[-1]]))

    ax.plot(pre_x, pre_y, 'r-')
    plt.show()

main()

```

Result:

![yy love](image/20170421.png)