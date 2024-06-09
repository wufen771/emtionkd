import sys
from matplotlib.pyplot import bar_label
sys.path.append("C:\\Users\\ff\\Desktop\\code\\EmotionKD-2\\source_code")
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from Buildmodel import get_GSR_model, get_MultiModal_model
from scipy.interpolate import interp1d
import numpy as np

def load_data(eeg_folder, gsr_folder, labels_folder, selected_participants):
    eeg_data = []
    gsr_data = []
    labels = []
    
    for participant_id in selected_participants:
        # 加载 EEG 数据
        eeg_file = os.path.join(eeg_folder, f"s{participant_id:02d}_eeg.csv")
        eeg_df = pd.read_csv(eeg_file, skiprows=1)
        # 使用插值填充缺失值
        eeg_df = eeg_df.interpolate()

        # 对每个样本进行插值，使得它们的长度都等于最大样本长度
        max_length = max(len(sample) for sample in eeg_df.values)
        interpolated_eeg_data = []
        for sample in eeg_df.values:
            # 生成新的索引，用于插值
            new_index = np.linspace(0, len(sample) - 1, max_length)
            # 进行线性插值
            f = interp1d(np.arange(len(sample)), sample, kind='linear')
            interpolated_sample = f(new_index)
            interpolated_eeg_data.append(interpolated_sample)

        eeg_data.append(interpolated_eeg_data)

        # 加载 GSR 数据
        gsr_file = os.path.join(gsr_folder, f"s{participant_id:02d}_gsr.csv")
        gsr_df = pd.read_csv(gsr_file, skiprows=1)
        # 使用插值填充缺失值
        gsr_df = gsr_df.interpolate()

        # 对每个样本进行插值，使得它们的长度都等于最大样本长度
        max_length = max(len(sample) for sample in gsr_df.values)
        interpolated_gsr_data = []
        for sample in gsr_df.values:
            # 生成新的索引，用于插值
            new_index = np.linspace(0, len(sample) - 1, max_length)
            # 进行线性插值
            f = interp1d(np.arange(len(sample)), sample, kind='linear')
            interpolated_sample = f(new_index)
            interpolated_gsr_data.append(interpolated_sample)

        gsr_data.append(interpolated_gsr_data)

        # 加载标签数据
        labels_file = os.path.join(labels_folder, f"participant_{participant_id}_labels.csv")
        labels_df = pd.read_csv(labels_file)
        # 使用插值填充缺失值
        labels_df = labels_df.interpolate()
        labels.append(labels_df[['valence', 'arousal']])

    return eeg_data, gsr_data, labels

def preprocess_data(eeg_data, gsr_data):
    # 对 EEG 数据进行标准化
    scaler_eeg = StandardScaler()
    eeg_data = [scaler_eeg.fit_transform(eeg) for eeg in eeg_data]

    # 对 GSR 数据进行标准化
    scaler_gsr = StandardScaler()
    gsr_data = [scaler_gsr.fit_transform(gsr) for gsr in gsr_data]

    return eeg_data, gsr_data

def adjust_sample_length(data):
    max_length = max(len(sample) for sample in data)
    adjusted_data = []
    for sample in data:
        if len(sample) < max_length:
            # 如果样本长度小于最大长度，则填充数据
            padding = max_length - len(sample)
            padded_sample = np.pad(sample, ((0, padding), (0, 0)), mode='constant')
            adjusted_data.append(padded_sample)
        elif len(sample) > max_length:
            # 如果样本长度大于最大长度，则截断数据
            truncated_sample = sample[:max_length, :]
            adjusted_data.append(truncated_sample)
        else:
            # 如果样本长度已经等于最大长度，则直接添加到调整后的数据中
            adjusted_data.append(sample)
    return np.array(adjusted_data)

# 定义文件夹路径
eeg_folder = "C:\\Users\\ff\\Desktop\\code\\deap_simple\\data_egg"
gsr_folder = "C:\\Users\\ff\\Desktop\\code\\deap_simple\\data_gsr"
labels_folder = "C:\\Users\\ff\\Desktop\\code\\deap_simple\\data_labels"

selected_participants = list(range(1,4))
# 加载数据
eeg_data, gsr_data, labels = load_data(eeg_folder, gsr_folder, labels_folder, selected_participants)


def smooth_labels(labels, window_size=5):
    smoothed_labels = []
    for i in range(len(labels)):
        start = max(0, i - window_size // 2)
        end = min(len(labels), i + window_size // 2)
        smoothed_label = sum(labels[start:end]) / (end - start)
        smoothed_labels.append(smoothed_label)
    return smoothed_labels

# 将数据按照8:1:1的比例划分为训练集、验证集和测试集
# 先将数据分成训练集和剩余的部分
# eeg_train_val, eeg_test, gsr_train_val, gsr_test, labels_train_val, labels_test = train_test_split(
#     eeg_data, gsr_data, labels, test_size=0.1, random_state=42)

# 再将剩余的部分分成验证集和测试集
# eeg_train, eeg_val, gsr_train, gsr_val, labels_train, labels_val = train_test_split(
#     eeg_train_val, gsr_train_val, labels_train_val, test_size=1/9, random_state=42)
# 数据分割
eeg_train, eeg_val, eeg_test = eeg_data
gsr_train, gsr_val, gsr_test = gsr_data
labels_train, labels_val, labels_test = labels
#输出每个数据集的大小
print("训练集大小:", len(eeg_train))
print("验证集大小:", len(eeg_val))
print("测试集大小:", len(eeg_test))
#打印加载的数据的前几行
print("EEG 数据示例:")
print(eeg_data[0])  # 打印第一个参与者的 EEG 数据
print("GSR 数据示例:")
print(gsr_data[0])  # 打印第一个参与者的 GSR 数据
print("标签数据示例:")
print(labels[0])    # 打印第一个参与者的标签数据



# 初始化 GSR 模型
gsr_model, gsr_custom_dir = get_GSR_model()
gsr_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
x_train=gsr_train
y_train=labels_train



# 初始化多模态模型
multi_modal_model, multi_modal_custom_dir = get_MultiModal_model()
# 准备训练数据
x_train_eeg = eeg_train  # EEG 训练数据
x_train_gsr = gsr_train  # GSR 训练数据
y_train_multi = labels_train  # 对应的标签数据


# 使用模型生成硬标签（预测结果）
hard_labels = gsr_model.predict(x_train)

# 生成软标签
# 例如，可以使用原始标签进行平滑处理（例如使用滑动窗口平均值等方法）
soft_labels = smooth_labels(y_train)
# 定义一个函数来提取统计特征
def extract_statistical_features(data):
    mean = np.mean(data, axis=1)
    std_dev = np.std(data, axis=1)
    max_val = np.max(data, axis=1)
    min_val = np.min(data, axis=1)
    return np.column_stack((mean, std_dev, max_val, min_val))

# 定义评估指标函数
def evaluate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1score = f1_score(y_true, y_pred, average='weighted')
    return accuracy, f1score

# 定义训练步骤
def train_step(model, optimizer, loss_object, X_train, y_train):
    with tf.GradientTape() as tape:
        predictions = model(X_train, training=True)
        loss = loss_object(y_train, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


# 定义训练函数
def train_model(model, optimizer, loss_object, X_train, y_train, num_epochs):
    for epoch in range(num_epochs):
        train_loss = train_step(model, optimizer, loss_object, X_train, y_train)
        predictions = model.predict(X_train)
        train_accuracy, train_f1score = evaluate_metrics(y_train, np.argmax(predictions, axis=1))
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train F1 Score: {train_f1score:.4f}")

# 提取统计特征作为软特征
soft_features_gsr = extract_statistical_features(gsr_data)

# 定义模型的优化器和损失函数
learning_rate = 1e-6
optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

# 训练模型
history_gsr=gsr_model.fit(x_train, y_train)

# 训练模型
history_multi = multi_modal_model.fit([x_train_eeg, x_train_gsr], [y_train_multi, None, None])

num_epochs = 10
train_model(gsr_model, optimizer, loss_object, [gsr_train, hard_labels, soft_labels, soft_features_gsr], labels_train, num_epochs=10)
train_model(multi_modal_model, x_train_eeg, x_train_gsr, y_train_multi, num_epochs=10, batch_size=32)