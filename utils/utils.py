import os
import shutil
import logging
import torch
import numpy as np
from decimal import *
import torchvision.utils as vutils
from utils.record_db import start_expr
_join = os.path.join


def save_model(log_dir, Net, current_result, mean_result, epoch):
    net_save_path = os.path.join(log_dir, 'net_params.pkl')
    torch.save({
        'epoch': epoch + 1,
        'state_dict': Net.state_dict(),
        'best_prec': net_save_path,
    }, net_save_path)
    if current_result < mean_result:
        net_save_path = os.path.join(log_dir, 'net_params_best.pkl')
        torch.save({
            'epoch': epoch + 1,
            'state_dict': Net.state_dict(),
            'best_prec': mean_result,
        }, net_save_path)
        current_result = mean_result
    return current_result

def adjust_learning_rate(optimizer, base_lr, max_iters,
                             cur_iters, power=0.9):
    lr = base_lr * ((1 - float(cur_iters) / max_iters) ** (power))
    optimizer.param_groups[0]['lr'] = lr
    return lr

def finetune_load(Net, pkl_path, is_parallel=False):
    # load params
    pretrained = torch.load(pkl_path)
    print('best=', pretrained['best_prec'])
    print('ep=', pretrained['epoch'])
    pretrained_dict = pretrained['state_dict']
    if is_parallel:
        pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}

    # 获取当前网络的dict
    net_state_dict = Net.state_dict()

    # 剔除不匹配的权值参数
    pretrained_dict_1 = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}

    # 更新新模型参数字典
    net_state_dict.update(pretrained_dict_1)

    # 将包含预训练模型参数的字典"放"到新模型中
    Net.load_state_dict(net_state_dict)

def myprint(logging, message):
    logging.info(message)
    print(message)

def print_writer_scalar(writer, logging, dict_message, step, mode):
    if mode == 'train':
        message = 'Step: %s  ' % step
    else:
        message = '[******] Step: %s  ' % step
    for key in dict_message:
        message += key + ':  %s  ' % str(Decimal(dict_message[key]).quantize(Decimal('0.000000')))
        writer.add_scalar('%s_result/%s' % (mode, key), dict_message[key], step)

    if mode == 'train':
        myprint(logging, message)
    else:
        myprint(logging, message+'\n')

def print_writer_scalars(writer, dict_message_train, dict_message_test, step):
    for key in dict_message_test:
        writer.add_scalars('all_result/%s' % key,
                           {'train_%s'%key:dict_message_train[key],
                            'test_%s'%key:dict_message_test[key]}, step)


def make_log(current_dir, result_dir, dataset_name, experment_name, fun):
    EXPR_ID: int = start_expr(experment_name, '', '', '')
    print('EXPR_ID', EXPR_ID)
    log_dir = os.path.join(current_dir, result_dir, dataset_name, experment_name + '%s' % EXPR_ID)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # current_dir = os.path.abspath(os.getcwd())

    # shutil.copytree(_join(current_dir, 'model'), _join(log_dir, 'model'))
    shutil.copy(_join(current_dir, '%s.py' % fun), _join(log_dir, '%s.py' % fun))

    logging.basicConfig(level=logging.INFO,
                        filename=_join(log_dir, 'new.log'),
                        filemode='w',
                        format='%(asctime)s - : %(message)s')

    return log_dir, logging

