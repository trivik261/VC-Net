from sklearn.metrics import roc_auc_score
import numpy as np
import torch

def metrics_test_drive_dice(predict_av, target_av, mask,  contain_classes):
    # 统计预测信息
    # F1A,F1V
    dice = []
    predict_av = predict_av[mask == 1].numpy()
    target_av = target_av[mask == 1].numpy()
    TP_AV = ((predict_av == contain_classes[1]) & (target_av == contain_classes[1])).sum()
    TN_AV = ((predict_av == contain_classes[0]) & (target_av == contain_classes[0])).sum()
    v_all = (predict_av == contain_classes[0]).sum() + (target_av == contain_classes[0]).sum()
    a_all = (predict_av == contain_classes[1]).sum() + (target_av == contain_classes[1]).sum()
    FA = TP_AV * 2 / a_all
    FV = TN_AV * 2 / v_all
    dice.append(FA)
    dice.append(FV)
    return dice

def metrics_test_drive_all(predict_av, target_av, predict_v, target_v, mask, contain_classes,smooth = 1e-5):
    # 统计预测信息
    #SEAV, SPAV, BACC
    predict_av = predict_av[mask == 1].numpy()
    target_av = target_av[mask == 1].numpy()
    TP_AV = ((predict_av == contain_classes[1]) & (target_av == contain_classes[1])).sum()
    TN_AV = ((predict_av == contain_classes[0]) & (target_av == contain_classes[0])).sum()
    FN_AV = ((predict_av == contain_classes[0]) & (target_av == contain_classes[1])).sum()
    FP_AV = ((predict_av == contain_classes[1]) & (target_av == contain_classes[0])).sum()
    sensitivity_av = TP_AV / (TP_AV + FN_AV)
    specificity_av = TN_AV / (TN_AV + FP_AV)
    balanced_accuracy_av = (sensitivity_av + specificity_av) / 2
    # accuracy_av = (TP_AV + TN_AV) / (TP_AV + TN_AV + FN_AV + FP_AV)

    # F1A,F1V
    accuracy = []
    dice = []
    v_all = (predict_av == contain_classes[0]).sum() + (target_av == contain_classes[0]).sum()
    a_all = (predict_av == contain_classes[1]).sum() + (target_av == contain_classes[1]).sum()
    FA = TP_AV * 2 / a_all
    FV = TN_AV * 2 / v_all
    dice.append(FA)
    dice.append(FV)

    #SE, SP, ACC
    pro = torch.clone(predict_v)
    predict_v[predict_v >= 0.5] = 1
    predict_v[predict_v < 0.5] = 0
    predict_v = predict_v[mask == 1].numpy()
    target_v = target_v[mask == 1].numpy()
    TP = ((predict_v == 1) & (target_v == 1)).sum()
    TN = ((predict_v == 0) & (target_v == 0)).sum()
    FN = ((predict_v == 0) & (target_v == 1)).sum()
    FP = ((predict_v == 1) & (target_v == 0)).sum()

    p = TP / (TP + FP)
    sensitivity_v = TP / (TP + FN)
    r = TP / (TP + FN)
    specificity_v = TN / (FP + TN)
    F1 = 2 * r * p / (r + p)
    accuracy_v = (TP + TN) / (TP + TN + FP + FN)
    # balanced_accuracy_v = (sensitivity_av + specificity_av) / 2

    #AUC
    pro = pro[mask == 1].numpy()
    auc = roc_auc_score(y_true=target_v, y_score=pro)

    return accuracy, dice, sensitivity_av, specificity_av, balanced_accuracy_av, \
           sensitivity_v, specificity_v, accuracy_v, auc, F1