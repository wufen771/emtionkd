import pickle

# 指定数据文件路径
data_file = 'C:\\Users\\ff\\Desktop\\code\\EmotionKD-2\\DEAP1\\data_preprocessed_python\\data_preprocessed_python\\s01.dat'

# 加载数据
with open(data_file, 'rb') as f:
    data = pickle.load(f, encoding='latin1')

# 打印字典的键
print(data.keys())
