import pandas as pd
import numpy as np
import os

from utils import metrics_evaluation, preprocess_data

train_rate = 0.8
seq_len = 12
pre_len = 3
 
# dataset = "metr_la"
# path = f"data/{dataset}/speed.csv"
# save_path = f"result_predict/{dataset}/{pre_len}/HA"

# Для данных microservices
dataset = "microservices"
adj = "good"
service = "no"
type_data = "calls"
path = f"data/{dataset}/{service}_fault_{type_data}.csv"
save_path = f"result_predict/{dataset}/{adj}/{service}/{type_data}/{pre_len}/HA"

if not os.path.exists(save_path):
    os.makedirs(save_path)

data = pd.read_csv(path)

time_len = data.shape[0]
num_nodes = data.shape[1]
trainX, trainY, testX, testY = preprocess_data(data, train_rate, seq_len, pre_len)

result = []
for sample in testX:
    sample_array = np.array(sample)
    tmp_result = []
    
    sample_mean_value = np.mean(sample_array, axis=0)
    tmp_result.append(sample_mean_value)
    
    for idx in range(1, pre_len):
        sample_array = sample_array[1:]
        sample_array = np.append(sample_array, [sample_mean_value], axis=0)
        sample_mean_value = np.mean(sample_array, axis=0)
        tmp_result.append(sample_mean_value)
    result.append(tmp_result)


result = np.array(result).transpose(0, 2, 1)
print(result.shape)
# np.save(f"{save_path}/true.npy", testY)
np.save(f"{save_path}/test.npy", result)

# result = np.reshape(result, [-1, num_nodes])
# testY = np.reshape(testY, [-1, num_nodes])
# rmse, mae, mape, r2 = metrics_evaluation(y_true=testY, y_pred=result)  
# print({
#     'RMSE': rmse, 
#     'MAE': mae,
#     'MAPE': mape,
#     'R2': r2,
# })
  