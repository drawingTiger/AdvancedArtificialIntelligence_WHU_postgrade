import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# 加载MNIST数据集
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist.data  # 特征 (784维，28x28的图像)
y = mnist.target.astype(np.uint8)  # 标签 (0-9)

# 数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 定义不同学习率和隐藏节点数
learning_rates = [0.1, 0.01, 0.001, 0.0001]
hidden_layer_sizes = [500, 1000,1500,2000]

# 存储结果
results = {}

# 训练模型并记录准确率、错误数量和正确数量
for hidden_size in hidden_layer_sizes:
    print(f"Training with hidden layer size: {hidden_size}")
    accuracies = []  # 存储当前隐藏节点数下的准确率
    error_counts = []  # 存储当前隐藏节点数下的错误数量
    correct_counts = []  # 存储当前隐藏节点数下的正确数量

    for lr in learning_rates:
        print(f"  Learning rate: {lr}")
        mlp = MLPClassifier(hidden_layer_sizes=(hidden_size,),  # 隐藏层大小
                            max_iter=100,                      # 最大迭代次数
                            learning_rate_init=lr,             # 初始学习率
                            random_state=42,                   # 随机种子
                            verbose=False)                     # 不打印训练过程

        # 训练模型
        mlp.fit(X_train, y_train)

        # 预测
        y_pred = mlp.predict(X_test)

        # 计算准确率
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)

        # 计算错误数量和正确数量
        errors = np.sum(y_test != y_pred)  # 错误分类的数量
        correct = np.sum(y_test == y_pred)  # 正确分类的数量
        error_counts.append(errors)
        correct_counts.append(correct)

        print(f"    Accuracy with learning rate {lr}: {acc:.4f}")
        print(f"    Error count with learning rate {lr}: {errors}")
        print(f"    Correct count with learning rate {lr}: {correct}")

    # 将当前隐藏节点数下的结果存储到结果中
    results[hidden_size] = {
        "accuracies": accuracies,
        "error_counts": error_counts,
        "correct_counts": correct_counts
    }

# 绘制不同隐藏节点数下的准确率、错误数量和正确数量折线图
plt.figure(figsize=(18, 12))

# 绘制准确率
plt.subplot(3, 1, 1)
for hidden_size, data in results.items():
    plt.plot(learning_rates, data["accuracies"], marker='o', label=f"Hidden Size: {hidden_size}")
plt.title("Accuracy vs Learning Rate for Different Hidden Layer Sizes")
plt.xlabel("Learning Rate")
plt.ylabel("Accuracy")
plt.xscale('log')  # 使用对数坐标轴
plt.legend()
plt.grid(True)

# 绘制错误数量
plt.subplot(3, 1, 2)
for hidden_size, data in results.items():
    plt.plot(learning_rates, data["error_counts"], marker='o', label=f"Hidden Size: {hidden_size}")
plt.title("Error Count vs Learning Rate for Different Hidden Layer Sizes")
plt.xlabel("Learning Rate")
plt.ylabel("Error Count")
plt.xscale('log')  # 使用对数坐标轴
plt.legend()
plt.grid(True)

# 绘制正确数量
plt.subplot(3, 1, 3)
for hidden_size, data in results.items():
    plt.plot(learning_rates, data["correct_counts"], marker='o', label=f"Hidden Size: {hidden_size}")
plt.title("Correct Count vs Learning Rate for Different Hidden Layer Sizes")
plt.xlabel("Learning Rate")
plt.ylabel("Correct Count")
plt.xscale('log')  # 使用对数坐标轴
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()