import csv
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# 数据分割
def data_partition(data_vectors):
    X_train, y_train, X_test, y_test = [], [], [], []
    i, j = 0, 0
    for item in data_vectors:
        if item[2] == [1, 0]:
            i += 1
            if i <= 201:
                X_test.append(item[1])
                y_test.append(item[2])
            else:
                X_train.append(item[1])
                y_train.append(item[2])
        else:
            j += 1
            if j <= 200:
                X_test.append(item[1])
                y_test.append(item[2])
            else:
                X_train.append(item[1])
                y_train.append(item[2])
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)


# 读取数据
def read_data(file1_path, file2_path):
    dataset1_reader = csv.reader(open(file1_path, encoding='utf-8'))
    dataset2_reader = csv.reader(open(file2_path, encoding='utf-8'))
    all_data_set = []
    for row in dataset1_reader:
        all_data_set.append([row[0], row[1], row[2], row[3]])
    for row in dataset2_reader:
        all_data_set.append([row[0], row[1], row[2], row[3]])
    random.seed(2)
    random.shuffle(all_data_set)
    return all_data_set


# 向量化数据
def vectorize_data(dataset):
    max_seq_len = max(len(item[1]) for item in dataset)
    for item in dataset:
        item[1] += "N" * (max_seq_len - len(item[1]))
    x_cast = {"A": np.array([1, 0, 0, 0, 0, 0, 0, 0]),
              "U": np.array([0, 1, 0, 0, 0, 0, 0, 0]),
              "T": np.array([0, 0, 0, 0, 0, 0, 0, 1]),
              "G": np.array([0, 0, 1, 0, 0, 0, 0, 0]),
              "C": np.array([0, 0, 0, 1, 0, 0, 0, 0]),
              "N": np.array([0, 0, 0, 0, 0, 0, 0, 0]),
              "(": np.array([0, 0, 0, 0, 1, 0, 0, 0]),
              ")": np.array([0, 0, 0, 0, 0, 1, 0, 0]),
              ".": np.array([0, 0, 0, 0, 0, 0, 1, 0])}
    y_cast = {"TRUE": [1, 0], "FALSE": [0, 1]}
    vectorized_dataset = []
    for item in dataset:
        data = [x_cast[char] for char in item[1]]
        vectorized_dataset.append([item[0], data, y_cast[item[2]]])
    return vectorized_dataset


# CNN 模型
class CNNMonoModel(nn.Module):
    def __init__(self, embedding_size, fc_size, num_classes, dropout_prob):
        super(CNNMonoModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(6, embedding_size), stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=(164, 1), stride=1)

        # 计算展平后的特征数量
        self.flattened_size = self._get_flattened_size((1, 180, embedding_size))

        self.fc1 = nn.Linear(self.flattened_size, fc_size)  # 调整输入维度
        self.fc2 = nn.Linear(fc_size, num_classes)
        self.dropout = nn.Dropout(dropout_prob)

    def _get_flattened_size(self, input_shape):
        """计算展平后的特征数量"""
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)  # 假设批量大小为1的虚拟输入
            x = self.pool(torch.relu(self.conv1(x)))
            return x.numel()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # 展平为二维
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# 模型训练与评估
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def evaluate_model(model, dataloader, device):
    model.eval()
    correct, total, loss = 0, 0, 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += (pred == target.argmax(dim=1)).sum().item()
            total += target.size(0)
    average_loss = loss / len(dataloader)  # 计算所有批次的平均损失
    return loss / total, correct / total, average_loss


# 训练和保存模型
if __name__ == "__main__":
    LR = 0.001
    TRAINING_ITER = 10000
    BATCH_SIZE = 18
    SEQUENCE_LENGTH = 180
    EMBEDDING_SIZE = 8
    FC_SIZE = 128
    NUM_CLASSES = 2
    DROPOUT_KEEP_PROB = 0.5

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

    # 准备数据集
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32).unsqueeze(1),
                                   torch.tensor(y_train, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32).unsqueeze(1),
                                  torch.tensor(y_test, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNMonoModel(EMBEDDING_SIZE, FC_SIZE, NUM_CLASSES, DROPOUT_KEEP_PROB).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # 训练模型
    for epoch in range(TRAINING_ITER):
        train_model(model, train_loader, optimizer, criterion, device)
        if epoch % 1000 == 0:
            train_loss, train_acc, average_loss = evaluate_model(model, train_loader, device)
            test_loss, test_acc, average_loss = evaluate_model(model, test_loader, device)
            print("第 {} 次迭代:".format(epoch + 1))
            print(f"交叉熵为 {average_loss}")
            print(f"Epoch {epoch + 1}/{TRAINING_ITER} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
            print("==================")

    # 保存模型
    model_path = "model/cnn_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存至 {model_path}")

    # 加载模型
    model = CNNMonoModel(EMBEDDING_SIZE, FC_SIZE, NUM_CLASSES, DROPOUT_KEEP_PROB).to(device)
    model.load_state_dict(torch.load("model/cnn_model.pth", map_location=device, weights_only=True))
    model.eval()  # 切换到评估模式
    print("模型已成功加载！")
    test_loss, test_acc, test_average_loss = evaluate_model(model, test_loader, device)
    print(f"加载模型后 - Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")



