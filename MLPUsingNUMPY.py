import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 加载MNIST数据集
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist.data.astype(np.float32)  # 特征 (784维，28x28的图像)
y = mnist.target.astype(np.int32)  # 标签 (0-9)

# 数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 将标签转换为one-hot编码
def to_one_hot(y, num_classes):
    return np.eye(num_classes)[y]

y_train_one_hot = to_one_hot(y_train, 10)
y_test_one_hot = to_one_hot(y_test, 10)

# 定义激活函数和其导数
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# 定义交叉熵损失函数和其导数
def cross_entropy_loss(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-9)) / y_true.shape[0]

def cross_entropy_loss_derivative(y_true, y_pred):
    return (y_pred - y_true) / y_true.shape[0]

# 定义Softmax函数
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # 防止数值不稳定
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# 定义全连接神经网络
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # 初始化权重和偏置
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        # 第一层：输入层到隐藏层
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = relu(self.z1)

        # 第二层：隐藏层到输出层
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = softmax(self.z2)

        return self.a2

    def backward(self, X, y_true, learning_rate):
        # 计算输出层的误差
        dL_dz2 = cross_entropy_loss_derivative(y_true, self.a2)

        # 更新第二层的权重和偏置
        dL_dW2 = np.dot(self.a1.T, dL_dz2)
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)
        self.W2 -= learning_rate * dL_dW2
        self.b2 -= learning_rate * dL_db2

        # 计算隐藏层的误差
        dL_da1 = np.dot(dL_dz2, self.W2.T)
        dL_dz1 = dL_da1 * relu_derivative(self.z1)

        # 更新第一层的权重和偏置
        dL_dW1 = np.dot(X.T, dL_dz1)
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)
        self.W1 -= learning_rate * dL_dW1
        self.b1 -= learning_rate * dL_db1

    def train(self, X, y_true, learning_rate, epochs):
        self.losses = []  # 记录每个epoch的损失
        for epoch in range(epochs):
            # 前向传播
            output = self.forward(X)

            # 计算损失
            loss = cross_entropy_loss(y_true, output)
            self.losses.append(loss)

            # 反向传播
            self.backward(X, y_true, learning_rate)

            # 每100个epoch打印一次损失
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)

# 定义超参数
learning_rates = [0.1, 0.01, 0.001, 0.0001]
hidden_sizes = [500, 1000, 1500, 2000]
epochs = 1000

# 存储结果
results = {}

# 遍历超参数
for hidden_size in hidden_sizes:
    print(f"Training with hidden layer size: {hidden_size}")
    accuracies = []
    error_counts = []
    loss_curves = []

    for lr in learning_rates:
        print(f"  Learning rate: {lr}")
        model = NeuralNetwork(input_size=784, hidden_size=hidden_size, output_size=10)
        model.train(X_train, y_train_one_hot, learning_rate=lr, epochs=epochs)

        # 测试模型
        y_pred = model.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        errors = np.sum(y_pred != y_test)  # 错误数量
        accuracies.append(accuracy)
        error_counts.append(errors)
        loss_curves.append(model.losses)  # 记录损失曲线

        print(f"    Test Accuracy with learning rate {lr}: {accuracy:.4f}")
        print(f"    Error count with learning rate {lr}: {errors}")

    # 记录结果
    results[hidden_size] = {
        "accuracies": accuracies,
        "error_counts": error_counts,
        "loss_curves": loss_curves
    }

# 绘制损失曲线
plt.figure(figsize=(18, 12))
for hidden_size, data in results.items():
    for i, lr in enumerate(learning_rates):
        plt.plot(data["loss_curves"][i], label=f"Hidden Size: {hidden_size}, LR: {lr}")

plt.title("Loss vs Epoch for Different Hidden Layer Sizes and Learning Rates")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()