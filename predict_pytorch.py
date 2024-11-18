import torch
import numpy as np
import sys
import getopt
from torch.nn.functional import softmax

# 模型参数
SEQUENCE_LENGTH = 180  # 输入序列长度
EMBEDDING_SIZE = 8     # 字符嵌入大小
x_cast = {"A": np.array([1, 0, 0, 0, 0, 0, 0, 0]),
          "U": np.array([0, 1, 0, 0, 0, 0, 0, 0]),
          "T": np.array([0, 0, 0, 0, 0, 0, 0, 1]),
          "G": np.array([0, 0, 1, 0, 0, 0, 0, 0]),
          "C": np.array([0, 0, 0, 1, 0, 0, 0, 0]),
          "N": np.array([0, 0, 0, 0, 0, 0, 0, 0]),
          "(": np.array([0, 0, 0, 0, 1, 0, 0, 0]),
          ")": np.array([0, 0, 0, 0, 0, 1, 0, 0]),
          ".": np.array([0, 0, 0, 0, 0, 0, 1, 0])}

# 序列预处理
def usage():
    print("用法: python predict_pytorch.py -s 预测的miRNA序列\n")
    print(
        "示例: python predict_pytorch.py -s CGGGGUGAGGUAGUAGGUUGUGUGGUUUCAGGGCAGUGAUGUUGCCCCUCGGAAGAUAACUAUACAACCUACUGCCUUCCCUG")

def seq_process(seq):
    seq = seq.replace(' ', '').replace("\n", "").upper()
    for base in seq:
        if base not in "AUGCT":
            print("请输入正确的碱基\n")
            usage()
            exit(1)
    seq = seq[:SEQUENCE_LENGTH] if len(seq) >= SEQUENCE_LENGTH else seq + "N" * (SEQUENCE_LENGTH - len(seq))
    return seq

def seq_vectorize(seq):
    vectorized_seq = [x_cast[char] for char in seq]
    vectorized_seq = np.array(vectorized_seq, dtype=np.float32).reshape(1, 1, SEQUENCE_LENGTH, EMBEDDING_SIZE)
    return torch.tensor(vectorized_seq)

# 定义 CNN 模型（与训练时的结构保持一致）
class CNNMonoModel(torch.nn.Module):
    def __init__(self, embedding_size, fc_size, num_classes, dropout_prob):
        super(CNNMonoModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 128, kernel_size=(6, embedding_size), stride=1, padding=0)
        self.pool = torch.nn.MaxPool2d(kernel_size=(164, 1), stride=1)

        # 动态计算展平后的特征数量
        self.flattened_size = self._get_flattened_size((1, SEQUENCE_LENGTH, embedding_size))

        self.fc1 = torch.nn.Linear(self.flattened_size, fc_size)  # 使用动态计算的大小
        self.fc2 = torch.nn.Linear(fc_size, num_classes)
        self.dropout = torch.nn.Dropout(dropout_prob)

    def _get_flattened_size(self, input_shape):
        """动态计算展平后的特征数量"""
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)  # 创建一个虚拟输入张量
            x = self.pool(torch.relu(self.conv1(x)))  # 通过卷积和池化
            return x.numel()  # 计算展平后的大小

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x).squeeze(-1).squeeze(-1)  # 移除多余维度
        x = x.view(x.size(0), -1)  # 展平为二维
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# 预测函数
def predict(model_path, sequence):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNMonoModel(EMBEDDING_SIZE, 128, 2, 0.5).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 序列处理和向量化
    processed_seq = seq_process(sequence)
    vectorized_seq = seq_vectorize(processed_seq).to(device)

    # 模型预测
    with torch.no_grad():
        output = model(vectorized_seq)
        probs = softmax(output, dim=1)
        prediction = torch.argmax(probs, dim=1).item()

    # 输出结果
    if prediction == 0:
        print("是人类序列。")
    elif prediction == 1:
        print("不是人类序列。")
    print(f"预测概率: {probs.cpu().numpy()}")

# 命令行解析
if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hs:", ["help", "sequence="])
    except getopt.GetoptError:
        print("使用错误！\n")
        usage()
        sys.exit(1)

    if len(opts) < 1:
        usage()
        sys.exit(1)

    for op, value in opts:
        if op in ("-s", "--sequence"):
            input_seq = value
        elif op in ("-h", "--help"):
            usage()
            sys.exit()

    print("预测结果:")
    print("您的输入序列:", input_seq)

    model_path = "model/cnn_model.pth"  # 替换为你的模型路径
    predict(model_path, input_seq)
