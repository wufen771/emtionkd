import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from Buildmodel import get_GSR_model, get_MultiModal_model
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras.callbacks import ReduceLROnPlateau
from Distill_zoo import feedback_Loss # type: ignore

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
def segment_eeg_signal(data, segment_length=512, num_segments=28):
    segmented_data = []
    missing_segments = []  # 用于记录缺失的片段
    for trial_index, trial in enumerate(data):
        trial_segments = []
        step_size = (trial.shape[1] - segment_length) // (num_segments - 1)
        for start in range(0, step_size * (num_segments - 1) + 1, step_size):
            segment = trial[:, start:start + segment_length]
            if segment.shape[1] == segment_length:
                trial_segments.append(segment)
            else:
                missing_segments.append((trial_index, start))
        if len(trial_segments) == num_segments:
            segmented_data.append(np.array(trial_segments))
        else:
            print(f"Skipping trial {trial_index} with insufficient segments: {len(trial_segments)}")
    return np.array(segmented_data), missing_segments

def segment_gsr_signal(data, segment_length=512):
    segmented_data = []
    missing_segments = []  # 用于记录缺失的片段
    for trial_index, trial in enumerate(data):
        segment = trial[:, :segment_length]
        if segment.shape[1] == segment_length:
            segmented_data.append(segment)
        else:
            missing_segments.append((trial_index, segment.shape[1]))
            print(f"Skipping GSR trial {trial_index} with insufficient length: {segment.shape[1]}")
    return np.array(segmented_data), missing_segments

def read_data(filename):
    with open(filename, 'rb') as file:
        x = pickle._Unpickler(file)
        x.encoding = 'latin1'
        data = x.load()
    return data

files = []
for n in range(1, 33): 
    s = ''
    if n < 10:
        s += '0'
    s += str(n)
    files.append(s)

labels = []
data = []
for i in files: 
    fileph = "C:\\Users\\ff\\Desktop\\code\\EmotionKD-2\\DEAP1\\data_preprocessed_python\\data_preprocessed_python\\s" + i + ".dat"
    d = read_data(fileph)
    labels.append(d['labels'])
    data.append(d['data'])

labels = np.array(labels)
data = np.array(data)
labels = labels.reshape(1280, 4)
data = data.reshape(1280, 40, 8064)
eeg_data = data[:,:32,:]
gsr_data = data[:,36:37,:]

# 对 EEG 数据进行分段
eeg_data_segmented,missing_eeg_segments = segment_eeg_signal(eeg_data)
print("Missing EEG Segments:")
for trial_index, start in missing_eeg_segments:
    print(f"Trial {trial_index}: Segment starting at {start}")

# 调整形状为 (1280, 28, 512, 1)
segmented_eeg_data = eeg_data_segmented.reshape(1280, 28, 512, 32)  # 确保形状为 (1280, 28, 512, 32)
segmented_eeg_data = np.mean(segmented_eeg_data, axis=-1)  # 对 32 个通道取平均值
segmented_eeg_data = np.expand_dims(segmented_eeg_data, axis=-1)  # 在最后添加一个维度

# 对 GSR 数据进行分段
gsr_data_segmented,missing_gsr_segments = segment_gsr_signal(gsr_data)
segmented_gsr_data = np.expand_dims(gsr_data_segmented, axis=-1)  # 在最后添加一个维度

print("Missing GSR Segments:")
for trial_index, length in missing_gsr_segments:
    print(f"Trial {trial_index}: Length {length}")

# 重新检查形状
print(segmented_eeg_data.shape)  # 应该是 (1280, 28, 512, 1)
print(segmented_gsr_data.shape)  # 应该是 (1280, 1, 512, 1)


df_label = pd.DataFrame({'Valence': labels[:,0], 'Arousal': labels[:,1], 
                        'Dominance': labels[:,2], 'Liking': labels[:,3]})
df_label.info()

label_name = ["valence","arousal","dominance","liking"]
labels_valence = []
labels_arousal = []
labels_dominance = []
labels_liking = []
for la in labels:
    if la[0] > 5:
        labels_valence.append(1)
    else:
        labels_valence.append(0)
    if la[1] > 5:
        labels_arousal.append(1)
    else:
        labels_arousal.append(0)
    if la[2] > 5:
        labels_dominance.append(1)
    else:
        labels_dominance.append(0)
    if la[3] > 6:
        labels_liking.append(1)
    else:
        labels_liking.append(0)

# 数据归一化
scaler = MinMaxScaler()
eeg_data_normalized = scaler.fit_transform(segmented_eeg_data.reshape(-1, 512)).reshape(segmented_eeg_data.shape)
gsr_data_normalized = scaler.fit_transform(segmented_gsr_data.reshape(-1, 512)).reshape(segmented_gsr_data.shape)

# 划分训练集、验证集和测试集
total_samples = eeg_data_normalized.shape[0]
train_size = int(total_samples * 0.8)
val_size = int(total_samples * 0.1)
test_size = total_samples - train_size - val_size

train_eeg = eeg_data_normalized[:train_size]
val_eeg = eeg_data_normalized[train_size:train_size + val_size]
test_eeg = eeg_data_normalized[train_size + val_size:]

train_gsr = gsr_data_normalized[:train_size]
val_gsr = gsr_data_normalized[train_size:train_size + val_size]
test_gsr = gsr_data_normalized[train_size + val_size:]

train_labels_valence = labels_valence[:train_size]
val_labels_valence = labels_valence[train_size:train_size + val_size]
test_labels_valence = labels_valence[train_size + val_size:]

# 转换为 one-hot 编码
train_labels_valence = tf.keras.utils.to_categorical(train_labels_valence, num_classes=2)
val_labels_valence = tf.keras.utils.to_categorical(val_labels_valence, num_classes=2)
test_labels_valence = tf.keras.utils.to_categorical(test_labels_valence, num_classes=2)

# 获取多模态模型
multi_modal_model = get_MultiModal_model()

# 训练多模态模型
multi_modal_model.fit([train_eeg, train_gsr], 
                      train_labels_valence, epochs=1, 
                      batch_size=32, 
                      validation_data=([val_eeg, val_gsr], val_labels_valence),
                      callbacks=[early_stopping, reduce_lr])

# 评估多模态模型
multi_modal_results = multi_modal_model.evaluate([test_eeg, test_gsr], test_labels_valence)
print("Multi-Modal Model Evaluation Results:", multi_modal_results)



# 从多模态模型生成软标签和注意力特征用于训练单模态模型
cls_train, _, att_features_train = multi_modal_model.predict([train_eeg, train_gsr])
cls_val, _, att_features_val = multi_modal_model.predict([val_eeg, val_gsr])
cls_test, _, att_features_test = multi_modal_model.predict([test_eeg, test_gsr])

multi_modal_pred_labels = np.argmax(cls_test, axis=1)
true_test_labels = np.argmax(test_labels_valence, axis=1)

# 计算多模态模型的F1得分
multi_modal_f1_score = f1_score(true_test_labels, multi_modal_pred_labels, average='weighted')

print("Multi-Modal Model F1 Score:", multi_modal_f1_score)

print("atttrain",att_features_train.shape)
print("attval", att_features_val.shape)
print("atttest",att_features_test.shape)

temperature = 4
alpha=0.5
# 使用Softmax函数将输出转换为概率分布
def softmax(logits, temperature):
    exp_logits = np.exp(logits / temperature)
    softmax_output = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    return softmax_output

# 计算软标签
soft_labels_train = softmax(cls_train, temperature)
soft_labels_val = softmax(cls_val, temperature)
soft_labels_test = softmax(cls_test, temperature)

# 打印软标签以验证
print("Soft Labels Train: ", soft_labels_train.shape)
print("Soft Labels Val: ", soft_labels_val.shape)
print("Soft Labels Test: ", soft_labels_test.shape)

# 确认 att_features 的形状
print("att_features_train shape:", att_features_train.shape)
print("att_features_val shape:", att_features_val.shape)
print("att_features_test shape:", att_features_test.shape)

# 获取单模态模型
gsr_model = get_GSR_model()
# 编译单模态模型
gsr_model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])

# 训练单模态模型
gsr_model.fit([train_gsr, train_labels_valence, soft_labels_train, att_features_train], 
              train_labels_valence, epochs=1, batch_size=32,
              validation_data=([val_gsr, val_labels_valence, soft_labels_val, att_features_val], 
                               val_labels_valence),
              callbacks=[early_stopping, reduce_lr])

# 评估单模态模型
results = gsr_model.evaluate([test_gsr, test_labels_valence, soft_labels_test, att_features_test], test_labels_valence)
print("Evaluation Results:", results)

# 获取单模态模型的预测结果
gsr_predictions = gsr_model.predict([test_gsr, test_labels_valence, soft_labels_test, att_features_test])
gsr_pred_labels = np.argmax(gsr_predictions, axis=1)

# 计算单模态模型的F1得分
gsr_f1_score = f1_score(true_test_labels, gsr_pred_labels, average='weighted')
print("GSR Model F1 Score:", gsr_f1_score)











