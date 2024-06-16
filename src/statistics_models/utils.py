import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score


def preprocess_data(data, train_rate=0.8, seq_len=12, pre_len=3):
    data_len = data.shape[0]
    train_size = int(data_len * train_rate)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    trainX, trainY, testX, testY = [], [], [], []
    for i in range(len(train_data) - int(seq_len + pre_len - 1)):
        a = train_data[i: i + seq_len + pre_len]
        trainX.append(a[:seq_len])
        trainY.append(a[seq_len:seq_len + pre_len])
    for i in range(len(test_data) - int(seq_len + pre_len - 1)):
        b = test_data[i: i + seq_len + pre_len]
        testX.append(b[0 : seq_len])
        testY.append(b[seq_len : seq_len + pre_len])
    return trainX, trainY, testX, testY
    

def metrics_evaluation(y_true, y_pred):
    rmse = np.round(np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred)), 4)
    mae = np.round(mean_absolute_error(y_true=y_true, y_pred=y_pred), 4)
    mape = np.round(mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred), 4)
    r2 = np.round(r2_score(y_true=y_true, y_pred=y_pred), 4)
    return rmse, mae, mape, r2