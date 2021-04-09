from tensorboardX import SummaryWriter
import torch.optim as optim
from torch import nn
import torch
import os
_join = os.path.join
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from utils.utils import save_model, finetune_load, make_log
# from utils.mydataset_drive import MyDataset
from utils.mydataset import MyDataset
from utils.data_flow import model_validate, model_train, model_validate_patch
from experiments.config import process_config

def main():
    # ------------------------------------ step 0 : 获取参数-----------------------------------------------
    cfg = process_config('./experiments/drive_av/standard.json')
    # ------------------------------------ step 1 : 加载数据-----------------------------------------------
    train_data = MyDataset(cfg.train_data_path[0],channel=cfg.channels, is_train=True, transform=None, input_size=512)
    test_data = MyDataset(cfg.test_data_path[0],channel=cfg.channels, is_train=False, transform=None)
    train_loader = torch.utils.data.DataLoader(train_data,batch_size=cfg.batchsize, shuffle=True, num_workers=2)
    validate_loader = torch.utils.data.DataLoader(test_data,batch_size=1, shuffle=False, num_workers=2)
    # ------------------------------------ step 2 : 定义网络-----------------------------------------------
    # Init model
    from model.VC_Net import VC_Net as net
    if cfg.finetune == True:
        Net = net(cfg.channels, cfg.num_class).cuda()
        finetune_load(Net, cfg.pkl_path, False)
    else:
        Net = net(cfg.channels, cfg.num_class, True).cuda()
    # ------------------------------------ step 3 : 定义损失函数和优化器 ------------------------------------
    criterion = nn.CrossEntropyLoss().cuda() # 选择损失函数
    criterion1 = nn.BCELoss().cuda()# 选择损失函数
    optimizer = optim.Adam(Net.parameters(), lr=cfg.lr,weight_decay=1e-5)    # 选择优化器
    # ------------------------------------ step 4 : 训练 --------------------------------------------------
    # ================================ ##        新建writer
    log_dir, cfg.logging = make_log('./', cfg.result_dir, cfg.dataset_name, cfg.experment_name, cfg.fun_main)
    cfg.writer = SummaryWriter(log_dir=log_dir)

    current_result = 0
    for epoch in range(cfg.max_epoch):
        model_train(cfg, Net, train_loader, criterion, criterion1, optimizer, epoch)
        if epoch!=0 and epoch%(cfg.frequency_show)==0:
            mean_accuracy_v = model_validate(cfg, Net, validate_loader, epoch)
            print(optimizer.param_groups[0]['lr'])
            # ================================ ##        模型保存
            current_result = save_model(log_dir, Net, current_result, mean_accuracy_v, epoch)
    # ================================ ##        关闭writer
    cfg.writer.close()


if __name__ == '__main__':
    main()