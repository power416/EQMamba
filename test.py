import torch
import numpy as np
from options import *
from models import *
from torch.utils.data import DataLoader
from dataloader import *


os.environ["CUDA_VISIBLE_DEVICES"] = '3'
opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'

def test(model):
    model.eval()

    torch.set_float32_matmul_precision('high') 

    test_loader = DataLoader(dataset=Dataset(opt.test_path, file_name='datas.npy'), batch_size=opt.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    # model = nn.DataParallel(model)
    model = model.to(opt.device)
    ckp = torch.load(opt.snapshots_folder + 'best_model.pt', map_location=opt.device, weights_only=True)
    # ckp = torch.load(opt.snapshots_folder + 'epoch_95.pth', map_location=opt.device)['model']
    model.load_state_dict(ckp)
    
    preds_o, trues_o, preds_a, trues_a, preds_d, trues_d = [], [], [], [], [], []

    with torch.no_grad():
        for iteration, (x_in, y_o, y_a, y_d) in enumerate(test_loader):

            x_in = x_in.to(opt.device)

            x_o, x_a, x_d = model(x_in)
            
            batch,_ = x_o.shape
            for i in range(batch):
                preds_o.append(x_o[i].cpu().detach().numpy().item())
                trues_o.append(y_o[i].cpu().detach().numpy().item())
                preds_a.append(x_a[i].cpu().detach().numpy().item())
                trues_a.append(y_a[i].cpu().detach().numpy().item())
                preds_d.append(x_d[i].cpu().detach().numpy().item())
                trues_d.append(y_d[i].cpu().detach().numpy().item())
            
            print(f'\rstep :{iteration + 1}/{len(test_loader)}', end='', flush=True)

    np.save(results_path + 'preds_o.npy', preds_o)
    np.save(results_path + 'trues_o.npy', trues_o)
    np.save(results_path + 'preds_a.npy', preds_a)
    np.save(results_path + 'trues_a.npy', trues_a)
    np.save(results_path + 'preds_d.npy', preds_d)
    np.save(results_path + 'trues_d.npy', trues_d)

model = EQMamba()
# model = model.to(opt.device)

results_path = './results/'
if not os.path.exists(results_path):
    os.mkdir(results_path)
test(model)
print('\n')
