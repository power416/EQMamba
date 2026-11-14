import numpy as np
import math

def cal(TP, FP, FN):

    precision = TP / (TP+FP+1e-07)
    recall = TP / (TP+FN+1e-07)
    f1 = (2*precision*recall)/(precision+recall+1e-07)

    return precision, recall, f1

def evalue(y_pred, y_true):
    length, = y_pred.shape

    TP = np.sum(y_pred*y_true)
    FP = np.sum(y_pred) - TP
    FN = np.sum(y_true) - TP
    TN = length - TP - FP - FN

    precision, recall, f1 = cal(TP, FP, FN)

    return TP, FP, FN, TN, precision, recall, f1

def r2_metric(preds, trues):
    SS_tot = np.sum((trues - np.mean(trues)) ** 2)
    SS_res = np.sum((trues - preds) ** 2)
    R2 = 1 - (SS_res / SS_tot)

    return R2

def mse_metric(preds, trues):
    mse = np.mean((preds - trues) ** 2)
    
    return mse

def mae_metric(preds, trues):
    mae = np.abs(preds - trues).mean()
    
    return mae

def mape_metric(preds, trues):
    errors = preds-trues
    real_mask = (1 - (trues == 0))
    loss = np.abs(errors/(np.abs(trues)+1e-7))
    loss *= real_mask
    non_zero_len = np.sum(real_mask)
    if non_zero_len == 0:
        mape = 0
    else:
        mape = np.sum(loss)/non_zero_len

    return mape

def calculate_metric(preds, trues):
    error = trues - preds
    mean = np.mean(error)
    std = np.std(error)

    r2 = r2_metric(preds, trues)
    mae = mae_metric(preds, trues)
    mape = mape_metric(preds, trues)

    return r2, mean, std, mae, mape

def calc_distance(lat_source, lon_source, lat_station, lon_station):
    R = 6378.1370

    lat_source = np.radians(lat_source)
    lon_source = np.radians(lon_source)
    lat_station = np.radians(lat_station)
    lon_station = np.radians(lon_station)

    dlat = lat_station - lat_source
    dlon = lon_station - lon_source

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat_source) * np.cos(lat_station) * np.sin(dlon / 2) ** 2
    )

    distance = 2 * R * np.arcsin(np.sqrt(a))

    return distance
