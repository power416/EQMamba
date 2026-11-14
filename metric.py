from functions import *
 
# 假设有两个数组：真实值和模型预测值

preds_o_path = './results/preds_o.npy'
trues_o_path = './results/trues_o.npy'
preds_a_path = './results/preds_a.npy'
trues_a_path = './results/trues_a.npy'
preds_d_path = './results/preds_d.npy'
trues_d_path = './results/trues_d.npy'

preds_o = np.load(preds_o_path, allow_pickle=True)
trues_o = np.load(trues_o_path, allow_pickle=True)
preds_a = np.load(preds_a_path, allow_pickle=True)
trues_a = np.load(trues_a_path, allow_pickle=True)
preds_d = np.load(preds_d_path, allow_pickle=True)
trues_d = np.load(trues_d_path, allow_pickle=True)

r2_o, mean_o, std_o, mae_o, mape_o = calculate_metric(preds_o, trues_o)
r2_a, mean_a, std_a, mae_a, mape_a = calculate_metric(preds_a, trues_a)
r2_d, mean_d, std_d, mae_d, mape_d = calculate_metric(preds_d, trues_d)

with open('pred_results.txt', 'w') as file:
    file.write('Longitude:'+ '\n')
    file.write('R2=%.4f, Mean=%.4f, Rmse=%.4f, Mae=%.4f, Mape=%.4f' % (r2_o, mean_o, std_o, mae_o, mape_o) + '\n\n')

    file.write('Latitude:'+ '\n')
    file.write('R2=%.4f, Mean=%.4f, Rmse=%.4f, Mae=%.4f, Mape=%.4f' % (r2_a, mean_a, std_a, mae_a, mape_a) + '\n\n')

    file.write('Depth:'+ '\n')
    file.write('R2=%.4f, Mean=%.4f, Rmse=%.4f, Mae=%.4f, Mape=%.4f' % (r2_d, mean_d, std_d, mae_d, mape_d) + '\n')
