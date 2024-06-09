import pickle
import numpy as np
import scipy.io

# 加载 .dat 文件
file_path = 'C:\\Users\\ff\\Desktop\\code\\EmotionKD-2\\DEAP1\\data_preprocessed_python\\data_preprocessed_python\\s01.dat'
with open(file_path, 'rb') as f:
    data = pickle.load(f, encoding='latin1')

# 提取数据
eeg_data = data['data']  # shape: (40 trials, 40 channels, 8064 samples)
labels = data['labels']  # shape: (40 trials, 4 labels)

# 打印数据结构
print("EEG data shape:", eeg_data.shape)
print("Labels shape:", labels.shape)

# 提取并打印第一个试验的EEG数据和标签
eeg_trial_1 = eeg_data[0]  # shape: (40 channels, 8064 samples)
label_trial_1 = labels[0]  # shape: (4,)
print("EEG trial 1 shape:", eeg_trial_1.shape)
print("Label trial 1:", label_trial_1)


# 加载数据
data = scipy.io.loadmat('C:\\Users\\ff\\Desktop\\code\\EmotionKD-2\\DEAP1\\data_preprocessed_matlab\\s01.mat')

# 读取EEG数据和标签
eeg_data = data['data']  # 形状为 (40 trials, 40 channels, 8064 samples)
labels = data['labels']  # 形状为 (40 trials, 4 labels)

# 打印数据形状
print(f'EEG Data shape: {eeg_data.shape}')
print(f'Labels shape: {labels.shape}')

# 查看第一个试验的EEG数据和标签
print(f'First trial EEG data (first 32 channels): {eeg_data[0, :32, :]}')
print(f'First trial labels: {labels[0]}')

def read_file_headers(file_path):
    encodings = ['utf-8', 'latin-1', 'gbk']  # 尝试多种编码格式
    with open(file_path, 'rb') as file:
        for encoding in encodings:
            try:
                first_line = file.readline().decode(encoding).strip()  # 尝试使用不同的编码格式解码第一行内容
                headers = first_line.split()  # 按空格分割成标题列表
                return headers
            except UnicodeDecodeError:
                continue  # 如果解码错误，则继续尝试下一个编码格式
    return None  # 如果所有编码格式都无法成功解码，则返回 None

# 示例使用
file_path = "C:\\Users\\ff\\Desktop\\code\\EmotionKD-2\\DEAP1\\data_preprocessed_python\\data_preprocessed_python\\s01.dat"  # 你的`.dat`文件路径
column_titles = read_file_headers(file_path)
if column_titles:
    print(column_titles)
else:
    print("Failed to read column titles.")



