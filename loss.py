import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
def Co_teaching(output1,output2,target_var,criterion1,rate_schedule,noise_flags):
    choose_loss1 = criterion1(output1, target_var)
    choose_loss2 = criterion1(output2, target_var)
    ind_sorted1 = torch.argsort(choose_loss1)
    ind_sorted2 = torch.argsort(choose_loss2)
    remember_rate = 1 - rate_schedule
    num_remember = int(remember_rate * ind_sorted1.shape[0])
    chosen_indices_1_clean = ind_sorted1[:num_remember]
    chosen_indices_2_clean = ind_sorted2[:num_remember]
    clean1, noisy1 = coteach_count_clean_noisy_indices(chosen_indices_1_clean, noise_flags)
    clean2, noisy2 = coteach_count_clean_noisy_indices(chosen_indices_2_clean, noise_flags)

    #干净样本交叉熵损失
    CE_loss_1=torch.mean(choose_loss1[chosen_indices_2_clean])
    CE_loss_2=torch.mean(choose_loss2[chosen_indices_1_clean])
    loss1=CE_loss_1
    loss2=CE_loss_2
    return loss1,loss2,clean1,clean2,noisy1,noisy2

def Co_teaching_plus(output1,output2,target_var,criterion1,rate_schedule,step,device):
    output1_sofmax = F.softmax(output1, dim=1)
    output2_sofmax = F.softmax(output2, dim=1)
    _, pred1 = torch.max(output1_sofmax.data, 1)
    _, pred2 = torch.max(output2_sofmax.data, 1)
    pred1, pred2 = pred1.cpu().numpy(), pred2.cpu().numpy()
    logical_disagree_id=np.zeros(target_var.size(), dtype=bool)
    disagree_id = []
    for idx, p1 in enumerate(pred1): 
        if p1 != pred2[idx]:
            disagree_id.append(idx) 
            logical_disagree_id[idx] = True
    _update_step = np.logical_or(logical_disagree_id, step < 5000).astype(np.float32)
    update_step = Variable(torch.from_numpy(_update_step)).to(device)
    if len(disagree_id) > 0:
        update_labels = target_var[disagree_id]
        update_outputs = output1[disagree_id] 
        update_outputs2 = output2[disagree_id] 
        
        loss_1, loss_2 = Co_teaching(update_outputs, update_outputs2, update_labels, criterion1, rate_schedule)
    else:
        update_labels = target_var
        update_outputs = output1
        update_outputs2 = output2

        cross_entropy_1 = F.cross_entropy(update_outputs, update_labels)
        cross_entropy_2 = F.cross_entropy(update_outputs2, update_labels)

        loss_1 = torch.sum(update_step*cross_entropy_1)/update_labels.size()[0]
        loss_2 = torch.sum(update_step*cross_entropy_2)/update_labels.size()[0]

        
    return loss_1, loss_2 

def JoCoR(output1,output2,target_var,criterion1,rate_schedule,epoch,co_lambda=0.8):#when symmetric noise-20%,50%,asymmetric-40,lambda is 0.9 for cifar10,noise-80% is 0.65,cifar100 is 0.85
    loss_pick_1 = criterion1(output1, target_var) * (1-co_lambda)
    loss_pick_2 = criterion1(output2, target_var) * (1-co_lambda)
    loss_pick = (loss_pick_1 + loss_pick_2 + co_lambda * kl_loss_compute(output1,output2,reduce=False) + co_lambda * kl_loss_compute(output2, output1, reduce=False)).cpu()
    
    ind_sorted = torch.argsort(loss_pick)
    
    remember_rate = 1 - rate_schedule
    
    num_remember = int(remember_rate * ind_sorted.shape[0])
    ind_update=ind_sorted[:num_remember]
    loss=torch.mean(loss_pick[ind_update])
        
    
    return loss, loss

def kl_loss_compute(pred, soft_targets, reduce=True):
    
    kl = F.kl_div(F.log_softmax(pred, dim=1),F.softmax(soft_targets, dim=1),reduce=False)
    

    if reduce:
        return torch.mean(torch.sum(kl, dim=1))
    else:
        return torch.sum(kl, 1)

def count_clean_noisy(mask, noise_flags):
    mask=mask.detach().cpu()
    mask = mask.bool()
    clean_in_mask = ((noise_flags == 1) & mask).sum().float()
    noisy_in_mask = ((noise_flags == 0) & mask).sum().float()
    return clean_in_mask, noisy_in_mask
def coteach_count_clean_noisy_indices(indices, noise_flags):
    indice=indices.detach().cpu()
    clean = (noise_flags[indice] == 1).sum().float()
    noisy = (noise_flags[indice] == 0).sum().float()
    return clean, noisy