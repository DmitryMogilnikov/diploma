import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import os

from utils import metrics_evaluation, preprocess_data
from sklearn.svm import SVR

train_rate = 0.8
seq_len = 12
pre_len = 12

# dataset = "seoul"
# path = f"data/{dataset}/speed.csv"
# save_path = f"result_predict/{dataset}/{pre_len}/SVR"
# model_save_path = f"models/{dataset}/{pre_len}/SVR"

# Для данных microservices
dataset = "microservices"
adj = "bad"
service = "no"
type_data = "duration"
path = f"data/{dataset}/{service}_fault_{type_data}.csv"
save_path = f"result_predict/{dataset}/{adj}/{service}/{type_data}/{pre_len}/SVR"
model_save_path = f"models/{dataset}/{adj}/no/{type_data}/{pre_len}/SVR"

if not os.path.exists(save_path):
    os.makedirs(save_path)
    
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

data = pd.read_csv(path)

time_len = data.shape[0]
num_nodes = data.shape[1]
trainX, trainY, testX, testY = preprocess_data(data, train_rate, seq_len, pre_len)

def train_model():
    for idx in tqdm(range(35, num_nodes)):
        array_data = np.mat(data)
        node_data = array_data[:, idx]
        a_X, a_Y, t_X, t_Y = preprocess_data(node_data, train_rate, seq_len, pre_len)
        a_X = np.reshape(a_X, [-1, seq_len])

        a_Y = np.reshape(a_Y, [-1, pre_len])
        a_Y = np.mean(a_Y, axis=1)

        t_X = np.reshape(t_X, [-1, seq_len])
        t_Y = np.reshape(t_Y, [-1, pre_len])    
        svr_model=SVR(kernel='linear')
        svr_model.fit(a_X, a_Y)
        with open(f"{model_save_path}/detector_{idx}.pkl", "wb") as f:
            pickle.dump(svr_model, f)


def predict_model():
    result = []
    for idx in range(0, num_nodes):
        array_data = np.mat(data)
        node_data = array_data[:, idx]
        a_X, a_Y, t_X, t_Y = preprocess_data(node_data, train_rate, seq_len, pre_len)
        t_X = np.reshape(t_X, [-1, seq_len])
        a_X = np.reshape(a_X, [-1, seq_len])
        svr_model = []
        with open(f"{model_save_path}/detector_{idx}.pkl", "rb") as f:
            while True:
                try:
                    svr_model.append(pickle.load(f))
                except EOFError:
                    break
        pre = svr_model[0].predict(a_X)
        pre = np.array(np.transpose(np.mat(pre)))
        pre = pre.repeat(pre_len, axis=1)
        result.append(pre)
    result = np.array(result).transpose(1, 0, 2)
    print(result.shape)
    np.save(f"{save_path}/train.npy", result)

# train_model()
predict_model()

# result = np.reshape(result, [num_nodes, -1])
# result = np.transpose(result)

# print(result.shape)

# testY = np.reshape(testY, [-1, num_nodes])
# rmse, mae, mape, r2 = metrics_evaluation(y_true=testY, y_pred=result)  
# print({
#     'RMSE': rmse, 
#     'MAE': mae,
#     'MAPE': mape,
#     'R2': r2,
# })