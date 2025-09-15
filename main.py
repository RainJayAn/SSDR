import argparse
import time
import torch
import matplotlib
matplotlib.use('Agg')
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch.nn import functional as F
from data.lib import choose_dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import torch.optim
import torch.utils.data
import numpy as np
import random
import resnet as resnet
from loss import Co_teaching,Co_teaching_plus
model_names = sorted(
    name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
    and name.startswith("resnet") and callable(resnet.__dict__[name]))

parser = argparse.ArgumentParser(description='CIFAR10/100')
parser.add_argument('--arch',
                    metavar='ARCH',
                    default='resnet32',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet32)')
parser.add_argument('-j',
                    '--workers',
                    default=8,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs',
                    default=150,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size',
                    default=128,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 512)')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=0.1,
                    type=float,
                    metavar='LR',
                    help='initial learning rate')
parser.add_argument('--momentum',
                    default=0.9,
                    type=float,
                    metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay',
                    '--wd',
                    default=1e-4,#1e-4
                    type=float,
                    metavar='W',
                    help='weight decay (default: 5e-4)')
parser.add_argument('--dataset',
                    help='',
                    default='cifar10',
                    choices=['cifar10','cifar100','kvasir'],
                    type=str)
parser.add_argument('--save-dir',
                    dest='save_dir',
                    help='The directory used to save the trained models',
                    default='../checkpoint/',
                    type=str)
parser.add_argument('--method',
                    help='method used for learning ',
                    default='ce',
                    choices=['ce','co-teaching','co-teach++','ours'],
                    type=str)

parser.add_argument('--alpha',
                    default=1.0,
                    type=float,
                    help='Coefficient of shannon entropy')
parser.add_argument('--beta',
                    default=1.2,
                    type=float,
                    help='Coefficient of JS')

parser.add_argument('--seed',
                    default=101,
                    type=int,
                    help='seed for validation data split')
parser.add_argument('--log_dir',
                    default='../runs/',
                    type=str,
                    )
parser.add_argument('--noise_rate',
                    default=0.2,
                    type=float,
                    help='0-1'
                    ) 
parser.add_argument('--noise_type',
                    default='symmetric',
                    type=str,
                    help='[pairflip, symmetric]'
                    )


args = parser.parse_args()
print(args)
cudnn.benchmark = False
random.seed(66)#66 #66
np.random.seed(66)
torch.manual_seed(66)
torch.cuda.manual_seed(66)
torch.cuda.manual_seed_all(66)

all_train_data1, all_train_data2,test_data,num_classes=choose_dataset(args.dataset,args.seed,args.noise_type,args.noise_rate)
def worker_init_fn(worker_id):
        random.seed(15 + worker_id)

train_loader = torch.utils.data.DataLoader(all_train_data1,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.workers,
    worker_init_fn=worker_init_fn,
    pin_memory=True)
if args.dataset=='cifar10' or args.dataset=='cifar100':
    b1=128
    b2=128
elif args.dataset=='kvasir':
    b1=128
    b2=128

val_loader = torch.utils.data.DataLoader(all_train_data2,
    batch_size=b1,
    shuffle=False,
    num_workers=args.workers,
    pin_memory=True)

test_loader = torch.utils.data.DataLoader(test_data,
    batch_size=b2,
    shuffle=False,
    num_workers=args.workers,
    pin_memory=True)

def main():
    
    name_model = args.log_dir +args.method+'/'+ args.arch + "_" + args.dataset + "_" +"noise"+"_"+str(args.noise_rate)+"_"+args.method+ "_"+args.noise_type+"_"+str(args.batch_size)+"_"+datestr()
    writer = SummaryWriter(log_dir=name_model, comment=name_model)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model1 = resnet.__dict__[args.arch](num_classes=num_classes)
    model2 = resnet.__dict__[args.arch](num_classes=num_classes)
    model1.to(device)
    model2.to(device)
    print('model over')

    criterion = nn.CrossEntropyLoss(reduction='none').to(device)
    
    optimizer1 = torch.optim.SGD(model1.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
       
    optimizer2 = torch.optim.SGD(model2.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)  
                                
    lr_scheduler1 = torch.optim.lr_scheduler.MultiStepLR(
            optimizer1, milestones=[80,120], last_epoch= -1) 
    lr_scheduler2 = torch.optim.lr_scheduler.MultiStepLR(
            optimizer2, milestones=[80,120], last_epoch= -1) 
    
    num_epoch = args.epochs
    rate_schedule = np.ones(args.epochs)*args.noise_rate
    rate_schedule[:15] = np.linspace(0, args.noise_rate**1,15)
    for epoch in range(0, num_epoch):
        # train for one epoch
        print('current lr1 {:.5e}'.format(optimizer1.param_groups[0]['lr']))
        print('current lr2 {:.5e}'.format(optimizer2.param_groups[0]['lr']))
       
        torch.autograd.set_detect_anomaly(True)
        train(model1,model2 ,criterion,optimizer1,optimizer2,rate_schedule, epoch,writer,device)
        
        lr_scheduler1.step()
        lr_scheduler2.step()
        print(f"Epoch [{epoch+1}/{num_epoch}], Method: {args.method}, Learning Rate: {optimizer1.param_groups[0]['lr']}")
        evaluate(args,model1,model2,writer,epoch,device)        
        torch.cuda.empty_cache()
def datestr():
    now = time.gmtime()
    return '{:02}_{:02}___{:02}_{:02}'.format(now.tm_mday, now.tm_mon, now.tm_hour, now.tm_min)
def train(model1,model2, criterion, optimizer1,optimizer2, rate_schedule,epoch,writer,device):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_1 = AverageMeter()
    losses_2 = AverageMeter()
    
    top1_1 = AverageMeter()
    top2_1 = AverageMeter()
    top3_1 = AverageMeter()
    top1_2 = AverageMeter()
    top2_2 = AverageMeter()
    top3_2 = AverageMeter()
    top1_1.reset()
    top2_1.reset()
    top3_1.reset()
    top1_2.reset()
    top2_2.reset()
    top3_2.reset()
    losses_1.reset()
    losses_2.reset()
    data_time.reset()
    batch_time.reset()
    # switch to train mode
    model1.train()
    model2.train()
    end = time.time()
    iter=0
    
    
    for i, (input, target,noise_flags,_) in enumerate(train_loader):
        
        # measure data loading time
        data_time.update(time.time() - end)
        input_var = input.to(device)
        target_var = target.to(device)
        
        output1 = model1(input_var)
        output2 = model2(input_var)
        
        if args.noise_rate!= 0.0:
            output_clean1,target_clean1,output_ns1,target_ns1=get_clean_noise(output1,target_var,noise_flags)
            prec_clean1=accuracy(output_clean1.detach(), target_clean1)[0]
            prec_noise1=accuracy(output_ns1.detach(), target_ns1)[0]
            output_clean2,target_clean2,output_ns2,target_ns2=get_clean_noise(output2,target_var,noise_flags)
            prec_clean2=accuracy(output_clean2.detach(), target_clean2)[0]
            prec_noise2=accuracy(output_ns2.detach(), target_ns2)[0]
        else:
            prec_clean1=prec_noise1=torch.tensor([0])
            prec_clean2=prec_noise2=torch.tensor([0])
            
        if args.method == 'ours':
            choose_loss1 = criterion(output1, target_var)
            choose_loss2 = criterion(output2, target_var)
            output1_sofmax=F.softmax(output1,dim=1)
            output2_sofmax=F.softmax(output2,dim=1)

            ind_sorted1 = torch.argsort(choose_loss1)
            ind_sorted2 = torch.argsort(choose_loss2)
            remember_rate = 1 - rate_schedule[epoch]
            num_remember = int(remember_rate * ind_sorted1.shape[0])
            chosen_indices_1_clean = ind_sorted1[:num_remember]
            chosen_indices_1_noise = ind_sorted1[num_remember:]
            chosen_indices_2_clean = ind_sorted2[:num_remember]
            chosen_indices_2_noise = ind_sorted2[num_remember:]
            
            # clean sample cross entropy
            CE_loss_1=torch.mean(choose_loss1[chosen_indices_2_clean])
            CE_loss_2=torch.mean(choose_loss2[chosen_indices_1_clean])
            # ambiguous sample selection
            output_1_noise=output1[chosen_indices_1_noise]
            output_2_noise=output2[chosen_indices_2_noise]

            output_1_noise_sofmax=F.softmax(output_1_noise,dim=1)
            output_2_noise_sofmax=F.softmax(output_2_noise,dim=1)

            normalized_entropy1_noise = compute_entropy(output_1_noise_sofmax)
            normalized_entropy2_noise = compute_entropy(output_2_noise_sofmax)
            
            easy_sample_1,hard_sample_1=select_hard_and_easy_samples(normalized_entropy1_noise,0.8)
            easy_sample_2,hard_sample_2=select_hard_and_easy_samples(normalized_entropy2_noise,0.8)

            ind_easy_1,ind_hard_1=chosen_indices_1_noise[easy_sample_1],chosen_indices_1_noise[hard_sample_1]
            ind_easy_2,ind_hard_2=chosen_indices_2_noise[easy_sample_2],chosen_indices_2_noise[hard_sample_2]

            # simple sample shannon entropy
            entropy_1=compute_entropy(output1_sofmax)
            entropy_2=compute_entropy(output2_sofmax)
            normalized_entropy1=entropy_1[ind_easy_2]
            normalized_entropy2=entropy_2[ind_easy_1]
            
            # ambiguous sample JS
            output1noise_sofmax=output1_sofmax.detach()
            output2noise_sofmax=output2_sofmax.detach()
            js_loss1=compute_js_divergence(output1_sofmax[ind_hard_2],output2noise_sofmax[ind_hard_2])
            js_loss2=compute_js_divergence(output1noise_sofmax[ind_hard_1],output2_sofmax[ind_hard_1])

            if epoch >=30:

                loss1=CE_loss_1+args.alpha*normalized_entropy1.mean()+args.beta*js_loss1
                loss2=CE_loss_2+args.alpha*normalized_entropy2.mean()+args.beta*js_loss2
            else:
                loss1=CE_loss_1
                loss2=CE_loss_2
        elif args.method == 'co-teaching':
            loss1,loss2,clean1,clean2,noisy1,noisy2=Co_teaching(output1,output2,target_var,criterion,rate_schedule[epoch],noise_flags)
        elif args.method == 'co-teach++':
            loss1,loss2=Co_teaching_plus(output1,output2,target_var,criterion,rate_schedule[epoch],epoch*i,device)
        
        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()
        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()
        output1 = output1.float()
        loss1 = loss1.float()
        output2 = output2.float()
        loss2 = loss2.float()
        # measure accuracy and record loss
        prec1 = accuracy(output1.detach(), target_var)[0]
        prec2 = accuracy(output2.detach(), target_var)[0]
        losses_1.update(loss1.item(), input.size(0))
        losses_2.update(loss2.item(), input.size(0))

        top1_1.update(prec1.item(), input.size(0))
        top2_1.update(prec_clean1.item(), input.size(0))
        top3_1.update(prec_noise1.item(), input.size(0))
        top1_2.update(prec2.item(), input.size(0))
        top2_2.update(prec_clean2.item(), input.size(0))
        top3_2.update(prec_noise2.item(), input.size(0))

        losses=(losses_1.avg+losses_2.avg)/2
        top1=(top1_1.avg+top1_2.avg)/2
        top2=(top2_1.avg+top2_2.avg)/2
        top3=(top3_1.avg+top3_2.avg)/2
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        iter+=1
        
        if i % 20 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss:.4f} \t'
                  'Prec@1 {top1:.3f}\t'
                  'Prec_clean {top2:.3f}\t'
                  'Prec_noise {top3:.3f}'.format(
                      epoch,
                      i,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      top1=top1,
                      top2=top2,
                      top3=top3))
    
    writer.add_scalars('ACC/', {'train_model1': top1_1.avg}, epoch)
    writer.add_scalars('ACC/', {'train_clean_model1': top2_1.avg}, epoch)
    writer.add_scalars('ACC/', {'train_noise_model1': top3_1.avg}, epoch)
    writer.add_scalars('ACC/', {'train_model2': top1_2.avg}, epoch)
    writer.add_scalars('ACC/', {'train_clean_model2': top2_2.avg}, epoch)
    writer.add_scalars('ACC/', {'train_noise_model2': top3_2.avg}, epoch)
        
def evaluate(args,model1,model2,writer,epoch,device):
    model1.eval()
    model2.eval()
    correct1 = 0
    correct2 = 0
    label=[]
    with torch.no_grad():
        for data, target,_,_ in test_loader:
            if torch.cuda.is_available():
                data, target = data.to(device), target.to(device)
            else:
                data, target = data.cpu(), target.cpu()
            
            output_nosoftmax_1 = model1(data)
            output1 = torch.nn.functional.softmax(output_nosoftmax_1, dim=1)
            output_nosoftmax_2 = model2(data)
            output2 = torch.nn.functional.softmax(output_nosoftmax_2, dim=1)
            pred1= output1.argmax(
                dim=1,
                keepdim=True)
            pred2 = output2.argmax(
                dim=1,
                keepdim=True)
            correct1 += pred1.eq(target.view_as(pred1)).sum().item()
            correct2 += pred2.eq(target.view_as(pred2)).sum().item()
        writer.add_scalars('ACC/', {'test_model1': round(correct1 /len(test_loader.dataset),4)}, epoch)
        writer.add_scalars('ACC/', {'test_model2': round(correct2 /len(test_loader.dataset),4)}, epoch)
    print('\nTest_model1 set: Accuracy: {:.2f}%   Test_model2 set: Accuracy: {:.2f}%   '.format(100. * correct1 /
                                                   len(test_loader.dataset),100. * correct2 /
                                                   len(test_loader.dataset) ))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    if batch_size == 0:
        return [torch.tensor(0.0) for _ in topk] 
    _, pred = output.topk(maxk, 1, True, True)
    
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def get_clean_noise(output,target,noise_flag):
    clean_indices = (noise_flag == 1).nonzero(as_tuple=True)[0]
    noise_indices = (noise_flag == 0).nonzero(as_tuple=True)[0]
    output_clean,output_noise=output[clean_indices],output[noise_indices]
    target_clean,target_noise=target[clean_indices],target[noise_indices]
    
    return output_clean,target_clean,output_noise,target_noise
def compute_kl_divergence(p, q):
    """
    计算 p 和 q 之间的 KL 散度
    :param p: 预测分布 (概率分布)
    :param q: 目标分布 (概率分布)
    :return: KL 散度
    """

    # 使用 F.kl_div 计算 KL 散度
    kl_loss = F.kl_div(torch.log(p + 1e-8), q,reduction='batchmean')  # 防止 log(0) 出现
    return kl_loss


def compute_js_divergence(p, q):
    """
    计算 p 和 q 之间的 JS 散度
    :param p: 预测分布 (概率分布)
    :param q: 目标分布 (概率分布)
    :return: JS 散度
    """
    # 计算混合分布 M
    m = 0.5 * (p + q)
    
    # 计算 KL(P || M) 和 KL(Q || M)
    kl_pm = compute_kl_divergence(p, m)
    kl_qm = compute_kl_divergence(q, m)
    
    # 计算 JS 散度
    js_loss = 0.5 * (kl_pm + kl_qm)
    
    return js_loss  
def select_hard_and_easy_samples(entropy, easy_ratio=0.8):
    
    
    sorted_entropy, sorted_indices = torch.sort(entropy) 
    
    easy_threshold_idx = int(len(sorted_entropy) * easy_ratio)
    
    easy_samples = sorted_indices[:easy_threshold_idx]
    hard_samples = sorted_indices[easy_threshold_idx:]
    
    return easy_samples, hard_samples

def compute_entropy(output_sofmax):
    classes = output_sofmax.size(1)  # 类别数
    log_probs = torch.log(output_sofmax + 1e-8)  # 防止对数为0，加一个小的平滑项
    entropy = -torch.sum(output_sofmax * log_probs, dim=1)  # 每个样本的熵
    normalized_entropy = entropy / torch.log(torch.tensor(classes, dtype=torch.float32))  # 归一化
    return normalized_entropy
def get_ll(preds, targets, **args):
    return np.log(1e-12 + preds[np.arange(len(targets)), targets]).mean()
if __name__ == '__main__':
    main()
