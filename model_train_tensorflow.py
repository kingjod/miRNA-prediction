import csv
import random
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import summary
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import numpy as np
import math


# 数据分割
def data_partition(data_vectors):
    X_train, y_train, X_test, y_test = [], [], [], []
    i, j = 0, 0
    for item in data_vectors:
        if item[2] == [1, 0]:
            i = i + 1
            if i <= 201:
                X_test.append(item[1])
                y_test.append(item[2])
            else:
                X_train.append(item[1])
                y_train.append(item[2])
        else:
            j = j + 1
            if j <= 200:
                X_test.append(item[1])
                y_test.append(item[2])
            else:
                X_train.append(item[1])
                y_train.append(item[2])

    return X_train, y_train, X_test, y_test


# 读取数据
def read_data(file1_path, file2_path):
    dataset1_reader = csv.reader(open(file1_path, encoding='utf-8'))
    dataset2_reader = csv.reader(open(file2_path, encoding='utf-8'))
    # 定义一个列表来存储数据
    all_data_set = []

    # 将数据读入列表（名称、序列、类别、二级结构）
    for row in dataset1_reader:
        all_data_set.append([row[0], row[1], row[2], row[3]])
    for row in dataset2_reader:
        all_data_set.append([row[0], row[1], row[2], row[3]])
    # 随机打乱数据集
    random.seed(2)
    random.shuffle(all_data_set)
    return all_data_set


# 获取权重变量
def get_weights_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    weights = tf.Variable(initial, name="weights")
    return weights


# 卷积和池化
def conv_and_pooling(input_tensor, filter_height, filter_width,
                     depth, conv_deep, layer_name):
    with tf.name_scope(layer_name):
        conv_weights = get_weights_variable([filter_height, filter_width, depth, conv_deep])
        conv_bias = tf.Variable(tf.constant(0.1, shape=[conv_deep]), name="bias")
        conv = tf.nn.conv2d(input_tensor, conv_weights, strides=STRIDES,
                            padding='SAME')
        conv_relu = tf.nn.relu(tf.nn.bias_add(conv, conv_bias))
        conv_relu_pool = tf.nn.max_pool(conv_relu, ksize=KSIZE,
                                        strides=STRIDES, padding='VALID')
        return conv_relu_pool


# 全连接输出
def fc_output_inference(input_tensor, fc_size, output_size, keep_prob):
    shape_list = input_tensor.get_shape().as_list()
    nodes = shape_list[1] * shape_list[2] * shape_list[3]
    reshaped = tf.reshape(input_tensor, [-1, nodes])

    # 第一个全连接层
    fc1_weights = get_weights_variable([nodes, fc_size])
    fc1_bias = tf.Variable(tf.constant(0.1, shape=[fc_size]))
    fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_bias)

    # 避免过拟合，使用dropout正则化
    fc1 = tf.nn.dropout(fc1, keep_prob)

    # 第二个全连接层（输出层）
    fc2_weights = get_weights_variable([fc_size, output_size])
    fc2_bias = tf.Variable(tf.constant(0.1, shape=[output_size]))
    output = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_bias)

    return output


# 单通道CNN推断
def cnn_mono_inference(input_tensor, filter_height, filter_width,
                       in_channels, out_channels, layer_name, keep_prob):
    # 卷积和最大池化层
    conv_pool = conv_and_pooling(input_tensor, filter_height,
                                 filter_width, 1, out_channels, layer_name)

    output = fc_output_inference(conv_pool, FC_SIZE, NUM_CLASSES, keep_prob)

    return output


# 模型输出
def model_output(input_X, EMBEDDING_SIZE, keep_prob):
    return cnn_mono_inference(input_X, 6, EMBEDDING_SIZE,
                              1, 128, "model-128", keep_prob)


# 评估操作
def evaluation_op(predic_output, input_ys):
    # 在测试数据上计算TP、TN、FP、FN
    # True Positive (TP):表示模型正确地将正类别样本预测为正类别。
    # True Negative (TN):表示模型正确地将负类别样本预测为负类别。
    # False Positive (FP):表示模型错误地将负类别样本预测为正类别。
    # False Negative (FN):表示模型错误地将正类别样本预测为负类别。
    predictions = tf.argmax(predic_output, 1)
    actuals = tf.argmax(input_ys, 1)

    ones_like_actuals = tf.ones_like(actuals)
    zeros_like_actuals = tf.zeros_like(actuals)
    ones_like_predictions = tf.ones_like(predictions)
    zeros_like_predictions = tf.zeros_like(predictions)

    tn_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, ones_like_actuals),
                tf.equal(predictions, ones_like_predictions)
            ),
            "float")
    )

    tp_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, zeros_like_actuals),
                tf.equal(predictions, zeros_like_predictions)
            ),
            "float")
    )

    fn_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, zeros_like_actuals),
                tf.equal(predictions, ones_like_predictions)
            ),
            "float")
    )

    fp_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, ones_like_actuals),
                tf.equal(predictions, zeros_like_predictions)
            ),
            "float")
    )
    return tp_op, tn_op, fp_op, fn_op


# 向量化数据
def vectorize_data(dataset):
    # 获取序列的最大长度
    max_seq_len = 0
    for item in dataset:
        if len(item[1]) > max_seq_len:
            max_seq_len = len(item[1])

    # 对最大序列长度进行填充
    for item in dataset:
        item[1] += "N" * (max_seq_len - len(item[1]))

    # one-hot编码
    x_cast = {"A": np.array([[1], [0], [0], [0], [0], [0], [0], [0]]),
              "U": np.array([[0], [1], [0], [0], [0], [0], [0], [0]]),
              "T": np.array([[0], [0], [0], [0], [0], [0], [0], [1]]),
              "G": np.array([[0], [0], [1], [0], [0], [0], [0], [0]]),
              "C": np.array([[0], [0], [0], [1], [0], [0], [0], [0]]),
              "N": np.array([[0], [0], [0], [0], [0], [0], [0], [0]]),
              "(": np.array([[0], [0], [0], [0], [1], [0], [0], [0]]),
              ")": np.array([[0], [0], [0], [0], [0], [1], [0], [0]]),
              ".": np.array([[0], [0], [0], [0], [0], [0], [1], [0]])}
    y_cast = {"TRUE": [1, 0], "FALSE": [0, 1]}  # TRUE: 人类基因序列  FALSE: 不是人类基因序列

    # 定义一个列表来存储向量化的数据
    vectorized_dataset = []

    for item in dataset:
        data = []
        for char in item[1]:
            data.append(x_cast[char])
        vectorized_dataset.append([item[0], data, y_cast[item[2]]])

    return vectorized_dataset


# 打印评估指标
# 打印评估指标
def print_evaluation(tp, tn, fp, fn):
    tpr = float(tp) / (float(tp) + float(fn))
    recall = tpr
    print("测试数据上的灵敏度/召回率为: {}".format(tpr))

    specifity = float(tn) / (float(tn) + float(fp))
    print("测试数据上的特异性为: {}".format(specifity))

    precision = float(tp) / (float(tp) + float(fp))
    print("测试数据上的精确度为: {}".format(precision))

    accuracy = (float(tp) + float(tn)) / (float(tp) + float(fp) + float(fn) + float(tn))
    print("测试数据上的准确度为: {}".format(accuracy))


# 训练模型
def train_model(log_path, model_path):
    cnn_output = model_output(input_X, EMBEDDING_SIZE, keep_prob)

    # 损失和优化
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=cnn_output, labels=input_y)
    cross_entropy_sum = tf.reduce_sum(cross_entropy)
    # 优化
    train_op = tf.train.AdamOptimizer(LR).minimize(cross_entropy_sum)

    # 计算模型准确度
    correct_prediction = tf.equal(tf.argmax(cnn_output, 1), tf.argmax(input_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    tf.summary.scalar("loss", cross_entropy_sum)
    tf.summary.scalar("accuracy", accuracy)
    # 在测试数据上的评估操作
    tp_op, tn_op, fp_op, fn_op = evaluation_op(cnn_output, input_y)

    merged = tf.summary.merge_all()
    saver = tf.train.Saver()
    tf.add_to_collection('pred_network', cnn_output)

    # 使用训练数据训练模型
    with tf.Session() as sess:  # 运行会话
        writer = tf.summary.FileWriter(log_path, sess.graph)
        sess.run(tf.global_variables_initializer())
        for i in range(TRAINING_ITER):
            start = (i * BATCH_SIZE) % dataset_size
            end = min(start + BATCH_SIZE, dataset_size)
            batch_xs = X_train[start:end]
            batch_ys = y_train[start:end]
            _, rs = sess.run([train_op, merged],
                             feed_dict={input_X: batch_xs, input_y: batch_ys,
                                        keep_prob: DROPOUT_KEEP_PROB})
            writer.add_summary(rs, i)

            if i % 1000 == 0:
                print("第 {} 次迭代:".format(i))
                print("交叉熵均值为：", end='')
                print(sess.run(cross_entropy_sum,
                               feed_dict={input_X: batch_xs, input_y: batch_ys, keep_prob: DROPOUT_KEEP_PROB}))

                print("在训练数据上的准确度:", end='')
                print(sess.run(accuracy, feed_dict={input_X: X_train,
                                                    input_y: y_train, keep_prob: 1}))

                print("在测试数据上的准确度:", end='')
                print(sess.run(accuracy, feed_dict={input_X: X_test,
                                                    input_y: y_test, keep_prob: 1}))
                print("==================")

            saver.save(sess, model_path)

        print("*********训练完成********")
        print("测试数据集性能：")
        tp, tn, fp, fn = sess.run([tp_op, tn_op, fp_op, fn_op],
                                  feed_dict={input_X: X_test,
                                             input_y: y_test, keep_prob: 1})
        print("tp:{}, tn:{}, fp:{}, fn:{}".format(tp, tn, fp, fn))
        print_evaluation(tp, tn, fp, fn)

        # 获取损失和准确度的历史数据
        loss_history = []
        accuracy_history = []
        for i in range(TRAINING_ITER):
            start = (i * BATCH_SIZE) % dataset_size
            end = min(start + BATCH_SIZE, dataset_size)
            batch_xs = X_train[start:end]
            batch_ys = y_train[start:end]
            loss_val, accuracy_val = sess.run([cross_entropy_sum, accuracy],
                                              feed_dict={input_X: batch_xs, input_y: batch_ys,
                                                         keep_prob: DROPOUT_KEEP_PROB})
            loss_history.append(loss_val)
            accuracy_history.append(accuracy_val)

        # 画图
        plt.figure(figsize=(10,5))

        plt.plot(range(0, len(loss_history), 1000), [loss_history[i] for i in range(0, len(loss_history), 1000)],
                 label='Loss')
        plt.title('Training Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('train_loss.png')


        plt.figure(figsize=(10,5))

        plt.plot(range(0, len(accuracy_history), 1000),
                 [accuracy_history[i] for i in range(0, len(accuracy_history), 1000)], label='Accuracy')
        plt.title('Training Accuracy')
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig('Accuracy.png')

        plt.show()



if __name__ == "__main__":
    LR = 0.001  # 学习率
    TRAINING_ITER = 10000  # 迭代次数
    BATCH_SIZE = 18  # 输入的批量大小

    SEQUENCE_LENGTH = 180  # 输入的序列长度
    EMBEDDING_SIZE = 8  # 字符嵌入大小（输入的序列宽度）

    STRIDES = [1, 1, 1, 1]  # 卷积过程中每个维度的步幅
    KSIZE = [1, 164, 1, 1]  # 池化窗口大小

    FC_SIZE = 128  # 全连接层的节点数
    NUM_CLASSES = 2  # 分类数

    DROPOUT_KEEP_PROB = 0.5  # 保持dropout的概率

    FILE_PATH = "data/true5.csv"
    FILE_PATH_PUTATIVE = "data/false5.csv"
    all_data_array = read_data(FILE_PATH, FILE_PATH_PUTATIVE)
    vectorized_dataset = vectorize_data(all_data_array)
    X_train, y_train, X_test, y_test = data_partition(vectorized_dataset)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    print(y_train.shape)
    print(X_train.shape)
    print(y_test.shape)
    print(X_test.shape)
    print("数据集向量化完成！")
    print("迭代次数", TRAINING_ITER)
    dataset_size = len(X_train)  # 训练数据集的数量

    input_X = tf.placeholder(tf.float32, [None, SEQUENCE_LENGTH, EMBEDDING_SIZE, 1], name='input_X')
    input_y = tf.placeholder(tf.float32, [None, NUM_CLASSES], name='input_y')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    train_model("model/model", "model/model/model.ckpt")
