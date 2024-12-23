import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import random
import pandas as pd  # 确保导入 pandas

# 加载MNIST数据集
mnist = fetch_openml('mnist_784', version=1, data_home='/ur/dataset/home/dir')
X, y = mnist["data"].astype(np.float32), mnist["target"].astype(np.float32)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=29)

# 测试不同 k 值的准确率
k_values = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]  # 测试 k 从 1 到 21 , 步长为 2
accuracies = []

for k in k_values:
    # 创建 KNN 分类器
    knn_clf = KNeighborsClassifier(n_neighbors=k)
    knn_clf.fit(X_train, y_train)

    # 预测
    y_pred = knn_clf.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

    # 计算预测正确的数量和预测错误的数量
    correct_count = np.sum(y_pred == y_test)
    error_count = len(y_test) - correct_count

    # 输出当前 k 值的结果
    print(f"k = {k}:  Accuracy: {accuracy:.4f},  Correct predictions: {correct_count},  Incorrect predictions: {error_count}")
    print("-" * 40)

# 绘制折线图
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='b')
plt.title('Accuracy vs. k in KNN', fontsize=16)
plt.xlabel('Number of Neighbors (k)', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.xticks(k_values)  # 设置 x 轴刻度为 k 的值
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# 使用最佳 k 值进行最终预测
best_k = k_values[np.argmax(accuracies)]
print(f"Best k value: {best_k}, Accuracy: {max(accuracies):.4f}")

knn_clf = KNeighborsClassifier(n_neighbors=best_k)
knn_clf.fit(X_train, y_train)
y_pred = knn_clf.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy with best k ({best_k}): {accuracy:.4f}")

# 随机展示部分预测结果
def plot_random_predictions(X_test, y_test, y_pred, num_samples=5):
    # 确保 X_test 是 numpy 数组
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values  # 转换为 numpy 数组
    elif isinstance(X_test, np.ndarray):
        pass  # 已经是 numpy 数组，无需转换
    else:
        raise ValueError("X_test 必须是 pandas.DataFrame 或 numpy.ndarray")

    # 确保 y_test 和 y_pred 是 numpy 数组
    if isinstance(y_test, pd.Series):
        y_test = y_test.values  # 转换为 numpy 数组
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values  # 转换为 numpy 数组

    # 检查 X_test 的形状是否正确
    if X_test.shape[1] != 784:
        raise ValueError("X_test 的列数必须是 784（28x28 图像的展开形式）")

    # 随机选择样本索引
    if num_samples > len(X_test):
        raise ValueError(f"num_samples 不能大于 X_test 的样本数（{len(X_test)}）")
    indices = random.sample(range(len(X_test)), num_samples)

    # 绘制图像
    plt.figure(figsize=(12, 6))
    for i, idx in enumerate(indices):
        plt.subplot(1, num_samples, i + 1)
        # 显示图像
        plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
        # 设置标题：预测值 (真实值)
        plt.title(f"Pred: {y_pred[idx]}\nTrue: {y_test[idx]}")
        plt.axis('off')
    plt.show()

# 调用函数展示随机预测结果
plot_random_predictions(X_test, y_test, y_pred)
