import glob
import numpy as np
import pandas as pd
from tensorflow import keras

from Buildmodel import get_GSR_model, get_MultiModal_model
import os
from sklearn.metrics import accuracy_score, f1_score

# Define the path to the folder containing the CSV files
gsr_folder_path = "C:\\Users\\ff\\Desktop\\code\\DEAP1\\data_gsr"
eeg_folder_path = "C:\\Users\\ff\\Desktop\\code\\DEAP1\\data_eeg"

# Get the list of CSV file paths
gsr_csv_files = glob.glob(gsr_folder_path + "s*_gsr.csv")

# Load the pre-trained GSR model and custom directory
gsr_model, gsr_custom_dir = get_GSR_model()

# Load the pre-trained MultiModal model and custom directory
multi_model, multi_custom_dir = get_MultiModal_model()

# Iterate over the GSR CSV files
for gsr_csv_file in gsr_csv_files:
    # Load the CSV file containing the raw GSR data
    gsr_data = pd.read_csv(gsr_csv_file)

    # Extract the GSR values from the data
    gsr_values = gsr_data["gsr"].values

    # Perform any necessary preprocessing on the GSR values (e.g., normalization, reshaping)

    # Reshape the GSR values to match the input shape of the GSR model
    gsr_input = gsr_values.reshape(1, 1, 512, 1)

    # Perform inference using the GSR model
    predictions_gsr = gsr_model.predict(gsr_input)

    print("File:", gsr_csv_file)
    # Print the predictions
    print(predictions_gsr)

# Iterate over the EEG CSV files
for eeg_csv_file in os.listdir(eeg_folder_path):
    if eeg_csv_file.endswith(".csv"):
        eeg_file_path = os.path.join(eeg_folder_path, eeg_csv_file)

        # Load the CSV file containing the raw EEG data
        eeg_data = pd.read_csv(eeg_file_path)

        # Extract the EEG values from the data
        eeg_values = eeg_data["eeg"].values

        # Perform any necessary preprocessing on the EEG values (e.g., normalization, reshaping)

        # Reshape the EEG values to match the input shape of the MultiModal model
        eeg_input = eeg_values.reshape(1, 28, 512, 1)

        # Perform inference using the MultiModal model
        predictions_eeg = multi_model.predict(eeg_input)

        # Print the predictions
        print("File:", eeg_csv_file)
        print("Predictions:", predictions_eeg)
        print("---")

# 定义文件夹路径
folder_path = "C:\\Users\\ff\\Desktop\\code\\DEAP1\\data_label"

# 初始化列表用于存储硬标签和软标签
hard_labels = []
soft_labels = []

# 遍历文件夹中的文件
for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv"):
        file_path = os.path.join(folder_path, file_name)

        # 读取标签文件
        df = pd.read_csv(file_path)
        
        # 提取硬标签和软标签（假设列名为 'valence' 和 'arousal'）
        hard_label = df[['valence', 'arousal']].values  # 硬标签
        soft_label = df.drop(columns=['valence', 'arousal']).values  # 软标签，去除了 'valence' 和 'arousal' 列

        # 将标签添加到列表中
        hard_labels.append(hard_label)
        soft_labels.append(soft_label)

# 打印硬标签和软标签的示例
print("Hard Labels (Example):", hard_labels[0])
print("Soft Labels (Example):", soft_labels[0])
# Assuming you have already defined and compiled your models, and prepared your data properly...

# Prepare training data
train_input_ori_eeg = eeg_input  # Training set original EEG data
train_input_ori_gsr = gsr_input  # Training set original GSR data
train_input_hard_label = hard_labels   # Training set hard labels

# Define hyperparameters
batch_size = 32
nums_epochs = 10

multi_model.fit(
    [train_input_ori_eeg, train_input_ori_gsr],
    [train_input_hard_label, None, None],
    batch_size=batch_size,
    epochs=10,
)

# Perform inference using MultiModal model
multi_predictions = multi_model.predict([train_input_ori_eeg, train_input_ori_gsr])
# Train GSR model
gsr_model.fit([train_input_ori_gsr, train_input_hard_label, train_input_soft_label, train_input_soft_feature],  epochs=10, batch_size=batch_size)
# Perform inference using GSR model
gsr_predictions = gsr_model.predict([train_input_ori_gsr, train_input_hard_label])
att_feature = gsr_predictions[2]
train_input_soft_label = ...  # Training set soft labels
train_input_soft_feature = att_feature  # Training set soft features
# Calculate evaluation metrics for MultiModal model
multi_accuracy = accuracy_score(hard_labels, multi_predictions)
multi_f1 = f1_score(hard_labels, multi_predictions, average="weighted")

# Calculate evaluation metrics for GSR model
gsr_accuracy = accuracy_score(hard_labels, gsr_predictions)
gsr_f1 = f1_score(hard_labels, gsr_predictions, average="weighted")

# Compare evaluation results
print("GSR Model - Accuracy:", gsr_accuracy)
print("GSR Model - F1 Score:", gsr_f1)
print("MultiModal Model - Accuracy:", multi_accuracy)
print("MultiModal Model - F1 Score:", multi_f1)
