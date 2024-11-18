import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import sys
import getopt
import numpy as np


def get_weights_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    weights = tf.Variable(initial, name="weights")
    return weights


SEQUENCE_LENGTH = 180  # 输入序列的长度
EMBEDDING_SIZE = 8  # 字符嵌入的大小（输入序列的宽度）

STRIDES = [1, 1, 1, 1]  # 卷积过程中每个维度的步幅
KSIZE = [1, 164, 1, 1]  # 池化窗口的大小

FC_SIZE = 128  # 全连接层的节点数
NUM_CLASSES = 2  # 分类数量

x_cast = {"A": np.array([[1], [0], [0], [0], [0], [0], [0], [0]]),
          "U": np.array([[0], [1], [0], [0], [0], [0], [0], [0]]),
          "T": np.array([[0], [0], [0], [0], [0], [0], [0], [1]]),
          "G": np.array([[0], [0], [1], [0], [0], [0], [0], [0]]),
          "C": np.array([[0], [0], [0], [1], [0], [0], [0], [0]]),
          "N": np.array([[0], [0], [0], [0], [0], [0], [0], [0]]),
          "(": np.array([[0], [0], [0], [0], [1], [0], [0], [0]]),
          ")": np.array([[0], [0], [0], [0], [0], [1], [0], [0]]),
          ".": np.array([[0], [0], [0], [0], [0], [0], [1], [0]])}


def usage():
    print("用法: python predict.py -s 预测的miRNA序列\n")
    print(
        "示例: python predict.py -s CGGGGUGAGGUAGUAGGUUGUGUGGUUUCAGGGCAGUGAUGUUGCCCCUCGGAAGAUAACUAUACAACCUACUGCCUUCCCUG")  # 是人类序列


def seq_process(seq):
    # 移除所有空格
    seq = seq.replace(' ', '')
    # 移除换行符
    seq = seq.replace("\n", "")
    for base in seq:
        if base not in ("A", "U", "G", "C", "T", "a", "u", "g", "c", "t"):
            print("请输入正确的碱基\n")
            usage()
            exit(1)
    seq = seq.upper()
    m_len = len(seq)
    if m_len < SEQUENCE_LENGTH:
        seq += "N" * (SEQUENCE_LENGTH - m_len)
    elif m_len >= SEQUENCE_LENGTH:
        seq = seq[:SEQUENCE_LENGTH]

    return seq


# 序列向量化
def seq_vectorize(seq):
    vectorized_seq = []
    for char in processed_seq:
        vectorized_seq.append(x_cast[char])
    vectorized_seq = np.array(vectorized_seq)
    vectorized_seq = vectorized_seq.reshape([1, SEQUENCE_LENGTH, EMBEDDING_SIZE, 1])
    return vectorized_seq


try:
    opts, args = getopt.getopt(sys.argv[1:], "hs:", ["help", "sequence="])
except getopt.GetoptError:
    print("使用错误！\n")
    usage()
    sys.exit(1)
if len(opts) < 1:
    usage()
    sys.exit(1)
# 解析选项
for op, value in opts:
    if op in ("-s", "--sequence"):
        input_seq = value
    elif op in ("-h", "--help"):
        usage()
        sys.exit()
print("预测结果:")
print("您的输入序列:", input_seq)
# 填充和向量化输入序列
processed_seq = seq_process(input_seq)
vectorized_seq = seq_vectorize(processed_seq)

# 导入已训练的模型参数
# 使用已训练的模型计算预测结果
with tf.Session() as sess:
    # 恢复已训练的模型
    saver = tf.train.import_meta_graph('model/model/model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint("model/model/"))

    predict_result = tf.get_collection('pred_network')[0]
    graph = tf.get_default_graph()

    # 通过get_operation_by_name()方法在恢复的图中获取占位符
    input_X = graph.get_operation_by_name('input_X').outputs[0]
    keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]

    # 提供新数据
    prediction = sess.run(predict_result, feed_dict={input_X: vectorized_seq, keep_prob: 1})

    # 打印预测类别
    m_index = sess.run(tf.argmax(prediction, 1))
    if m_index == 0:
        print("是人类序列。")
    elif m_index == 1:
        print("不是人类序列。")
