import os
import pyedflib
import pandas as pd

#把bdf转换成csv
def bdf_to_csv(dat_file, csv_file):
    # 打开BDF文件
    f = pyedflib.EdfReader(dat_file)
    # 获取信号的名称
    signal_labels = f.getSignalLabels()
    # 创建一个空的DataFrame来存储数据
    df = pd.DataFrame(columns=signal_labels)
    # 将数据读取到DataFrame中
    for i in range(len(signal_labels)):
        df[signal_labels[i]] = f.readSignal(i)
    # 关闭BDF文件
    f.close()
    # 将DataFrame保存为CSV文件
    df.to_csv(csv_file, index=False)
# 指定包含BDF文件的文件夹路径
folder_path = "C:\\Users\\ff\Desktop\\code\\EmotionKD-2\\DEAP1\\original"
csv_folder = "C:\\Users\\ff\\Desktop\\code\\EmotionKD-2\\DEAP1\\data_preprocessed_csv"
# 遍历文件夹中的所有文件
for file in os.listdir(folder_path):
    if file.endswith(".dat"):
        dat_file = os.path.join(folder_path, file)
        csv_file = os.path.join(csv_folder, file.replace(".dat", ".csv"))
        bdf_to_csv(dat_file, csv_file)

"""def extract_eeg_gsr_labels(input_folder, output_folder):
    # 遍历DEAP数据集中的每个CSV文件
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".csv"):
            csv_file_path = os.path.join(input_folder, file_name)
            
            # 读取CSV文件
            df = pd.read_csv(csv_file_path)
            
            # 提取EEG和GSR数据
            eeg_data = df.iloc[:, :32]  # 提取前32列为EEG数据
            gsr_data = df.iloc[:, 40:42]  # 提取第41-42列为GSR数据
            
            # 提取情感价值和情感激励标签
            valence_arousal_labels = df[['valence', 'arousal']]
            
            # 构造保存EEG、GSR和标签数据的文件路径
            eeg_output_file = os.path.join(output_folder, file_name.replace(".csv", "_eeg.csv"))
            gsr_output_file = os.path.join(output_folder, file_name.replace(".csv", "_gsr.csv"))
            labels_output_file = os.path.join(output_folder, file_name.replace(".csv", "_labels.csv"))
            
            # 保存EEG、GSR和标签数据到对应的CSV文件中
            eeg_data.to_csv(eeg_output_file, index=False)
            gsr_data.to_csv(gsr_output_file, index=False)
            valence_arousal_labels.to_csv(labels_output_file, index=False)

# 指定DEAP数据集的输入文件夹和输出文件夹
input_folder = "C:\\Users\\ff\\Desktop\\code\\DEAP1\\data_csv"
output_folder = "C:\\Users\\ff\\Desktop\\code\\DEAP1\\data_csv"

# 调用函数提取EEG、GSR和标签数据，并保存到对应的CSV文件夹中
extract_eeg_gsr_labels(input_folder, output_folder)


#读取 DEAP 数据集中的 CSV 文件
csv_file = "C:\\Users\\ff\\Desktop\\code\\DEAP1\\metadata_csv\\participant_ratings.csv"
df = pd.read_csv(csv_file)

# 遍历每个参与者
for participant_id in range(1, 33):
    # 提取该参与者的情感价值和情感激励标签
    participant_data = df[df['Participant_id'] == participant_id]
    participant_labels = participant_data[['Valence', 'Arousal']]
    participant_labels.columns = ['valence', 'arousal']
    
    # 构造保存标签数据的文件路径
    output_file = f"C:\\Users\\ff\\Desktop\\code\\DEAP1\\data_label\\participant_{participant_id}_labels.csv"
    
    # 将标签数据保存到对应的 CSV 文件中，每个文件包含 40 条信息
    participant_labels.to_csv(output_file, index=False)
"""
