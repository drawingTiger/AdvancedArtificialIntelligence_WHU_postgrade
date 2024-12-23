import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

# 1. 数据准备
# 加载MNIST数据集
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# 数据预处理：归一化到[0, 1]范围
train_images, test_images = train_images / 255.0, test_images / 255.0

# 将图像数据扩展为4D张量 (batch_size, height, width, channels)
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

# 2. 构建更深的CNN模型
model = models.Sequential([
    # 第一组卷积层和池化层
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

    # 第二组卷积层和池化层
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # 第三组卷积层和池化层
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # 展平层
    layers.Flatten(),

    # 全连接层
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),  # 防止过拟合

    # 输出层
    layers.Dense(10, activation='softmax')
])

# 3. 编译模型
# 使用Adam优化器和交叉熵损失函数
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. 训练模型
history = model.fit(train_images, train_labels, epochs=100, batch_size=64, validation_split=0.1)

# 5. 评估模型
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}")

# 6. 展示训练集和验证集的最终准确率和损失
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]

print(f"Final training accuracy: {final_train_acc:.4f}")
print(f"Final validation accuracy: {final_val_acc:.4f}")
print(f"Final training loss: {final_train_loss:.4f}")
print(f"Final validation loss: {final_val_loss:.4f}")

# 7. 绘制准确率和损失变化曲线
# 提取训练历史
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# 绘制准确率曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Accuracy Curves')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# 绘制损失曲线
plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

# 8. 展示测试集的预测结果
# 随机选择5个测试样本
random_indices = np.random.choice(test_images.shape[0], 5, replace=False)
sample_images = test_images[random_indices]
sample_labels = test_labels[random_indices]

# 预测样本
predictions = model.predict(sample_images)
predicted_labels = np.argmax(predictions, axis=1)

# 展示样本及其预测结果
plt.figure(figsize=(12, 6))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(sample_images[i].reshape(28, 28), cmap='gray')
    plt.title(f"True: {sample_labels[i]}\nPred: {predicted_labels[i]}")
    plt.axis('off')

plt.show()