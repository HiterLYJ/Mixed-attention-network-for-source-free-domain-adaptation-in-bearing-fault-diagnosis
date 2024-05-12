import argparse
import os
import os.path as osp

import numpy as np
import numpy.random as random
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader, Dataset

import loss
import network


# 可视化
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE


# 混淆矩阵
from pycm import *

# 关闭警告
import warnings

import pandas as pd




def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.5):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


class DataLoadStrain(Dataset):
    def __init__(self, args):
        SetofData = args.dset
        SchofData = args.dschool
        Source = SetofData[0:2]
        DicS = './data/' + SchofData + '/' + Source
        Sdata = scipy.io.loadmat(DicS + '/' + 'traindata.mat')['traindata']
        SLabl = scipy.io.loadmat(DicS + '/' + 'trainlabel.mat')['trainlabel']
        SLabl = torch.from_numpy(SLabl)
        SLabl = SLabl.long()
        Sdata = torch.from_numpy(Sdata)
        Sdata = Sdata.to(torch.float32)
        self.x_data = torch.unsqueeze(Sdata, dim=1)
        self.y_data = SLabl
        self.len = Sdata.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index], index

    def __len__(self):
        return self.len


class DataLoadTtrain(Dataset):
    def __init__(self, args):
        SetofData = args.dset
        SchofData = args.dschool
        Target = SetofData[2:4]
        DicT = './data/' + SchofData + '/' + Target
        Tdata = scipy.io.loadmat(DicT + '/' + 'traindata.mat')['traindata']
        TLabl = scipy.io.loadmat(DicT + '/' + 'trainlabel.mat')['trainlabel']
        TLabl = torch.from_numpy(TLabl)
        TLabl = TLabl.long()
        Tdata = torch.from_numpy(Tdata)
        Tdata = Tdata.to(torch.float32)
        self.x_data = torch.unsqueeze(Tdata, dim=1)
        self.y_data = TLabl
        self.len = Tdata.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index], index

    def __len__(self):
        return self.len


class DataLoadStest(Dataset):
    def __init__(self, args):
        SetofData = args.dset
        SchofData = args.dschool
        Source = SetofData[0:2]
        DicS = './data/' + SchofData + '/' + Source
        Sdata = scipy.io.loadmat(DicS + '/' + 'testdata.mat')['testdata']
        SLabl = scipy.io.loadmat(DicS + '/' + 'testlabel.mat')['testlabel']
        SLabl = torch.from_numpy(SLabl)
        SLabl = SLabl.long()
        Sdata = torch.from_numpy(Sdata)
        Sdata = Sdata.to(torch.float32)
        self.x_data = torch.unsqueeze(Sdata, dim=1)
        self.y_data = SLabl
        self.len = Sdata.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index], index

    def __len__(self):
        return self.len


class DataLoadTtest(Dataset):
    def __init__(self, args):
        SetofData = args.dset
        SchofData = args.dschool
        Target = SetofData[2:4]
        DicT = './data/' + SchofData + '/' + Target
        Tdata = scipy.io.loadmat(DicT + '/' + 'testdata.mat')['testdata']
        TLabl = scipy.io.loadmat(DicT + '/' + 'testlabel.mat')['testlabel']
        TLabl = torch.from_numpy(TLabl)
        TLabl = TLabl.long()
        Tdata = torch.from_numpy(Tdata)
        Tdata = Tdata.to(torch.float32)
        self.x_data = torch.unsqueeze(Tdata, dim=1)
        self.y_data = TLabl
        self.len = Tdata.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index], index

    def __len__(self):
        return self.len


def digit_load(args):
    train_bs = args.batch_size
    # 数据文件转换成两个数据集
    train_source = DataLoadStrain(args)
    train_target = DataLoadTtrain(args)
    test_source = DataLoadStest(args)
    test_target = DataLoadTtest(args)

    dset_loaders = {}
    # 2、5是两倍batch
    dset_loaders["source_tr"] = DataLoader(train_source, batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                           drop_last=False)
    dset_loaders["source_te"] = DataLoader(test_source, batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                           drop_last=False)
    # 一定要打乱
    # 分类均衡的时候用
    dset_loaders["target"] = DataLoader(train_target, batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                        drop_last=False)
    dset_loaders["target_te"] = DataLoader(train_target, batch_size=train_bs*2, shuffle=False, num_workers=args.worker,
                                           drop_last=False)
    dset_loaders["test"] = DataLoader(test_target, batch_size=train_bs, shuffle=False, num_workers=args.worker,
                                      drop_last=False)
    dset_loaders["visualizecf"] = DataLoader(test_target, batch_size=1, shuffle=False, num_workers=args.worker,
                                      drop_last=False)
    dset_loaders["visualizetsneT"] = DataLoader(test_target, batch_size=64, shuffle=False, num_workers=args.worker,
                                           drop_last=False)
    dset_loaders["visualizetsneS"] = DataLoader(test_source, batch_size=64, shuffle=False, num_workers=args.worker,
                                                drop_last=False)
    return dset_loaders


def cal_acc(loader, netF, netB, netC):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            # inputs = inputs
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == torch.squeeze(all_label)).item() / float(all_label.size()[0])
    # 没有用到过第二个输出
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()
    return accuracy * 100, mean_ent


def train_source(args):
    # plt.figure(num=0, figsize=(10,10))
    # fig1 = plt.figure(num=1)
    # plt.ion()
    dset_loaders = digit_load(args)
    netF = network.DTNBase().cuda()
    # netF = rn.RestNet18()
    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()
    epoch = np.array([0])
    acccc = np.array([0])
    param_group = []
    learning_rate = args.lr
    # named_parameters()输出神经网络的网络层和参数迭代器
    # parameters()只能输出神经网络的参数迭代器
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    acc_init = 0
    # 迭代次数
    max_iter = args.max_epoch * len(dset_loaders["source_tr"])
    # interval_iter = max_iter // 10
    interval_iter = len(dset_loaders["source_tr"])
    iter_num = 0

    netF.train()
    netB.train()
    netC.train()

    while iter_num < max_iter:
        try:
            inputs_source, labels_source = iter_source.next()
        except:
            iter_source = iter(dset_loaders["source_tr"])
            inputs_source, labels_source, _ = iter_source.next()

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()

        outputs_source = netC(netB(netF(inputs_source)))
        print(outputs_source.size())
        print(labels_source.size())
        classifier_loss = loss.CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_source,
                                                                                                        labels_source)
        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            print('《Target Only》 Loss = {:.2f}'.format(classifier_loss))
            netF.eval()
            netB.eval()
            netC.eval()
            acc_s_tr, _ = cal_acc(dset_loaders['source_tr'], netF, netB, netC)
            acc_s_te, _ = cal_acc(dset_loaders['source_te'], netF, netB, netC)
            log_str = 'Task: {}; Iter:{}/{}; Train Accuracy on Source = {:.2f}%; Test Accuracy on Source = {:.2f}%'.format(
                args.dset, iter_num, max_iter, acc_s_tr, acc_s_te)
            epoch = np.append(epoch, iter_num / interval_iter)
            acccc = np.append(acccc, acc_s_te)
            # fig1.clf()
            # plt.plot(epoch, acccc, lw=4, ls='-', c='b', alpha=0.1)
            # plt.title('Source')
            # plt.xlabel('Epoch')
            # plt.ylabel('Accuracy')
            # plt.pause(0.1)
            # plt.draw()
            # plt.ioff()
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')
            if acc_s_te >= acc_init:
                acc_init = acc_s_te
                best_netF = netF.state_dict()
                best_netB = netB.state_dict()
                best_netC = netC.state_dict()
            netF.train()
            netB.train()
            netC.train()
    torch.save(best_netF, osp.join(args.output_dir, "source_F.pt"))
    torch.save(best_netB, osp.join(args.output_dir, "source_B.pt"))
    torch.save(best_netC, osp.join(args.output_dir, "source_C.pt"))
    return netF, netB, netC


def target_test(args):
    dset_loaders = digit_load(args)
    netF = network.DTNBase().cuda()
    # netF = rn.RestNet18()
    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    args.modelpath = args.output_dir + '/source_F.pt'
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/source_B.pt'
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/source_C.pt'
    netC.load_state_dict(torch.load(args.modelpath))
    netF.eval()
    netB.eval()
    netC.eval()
    acc, _ = cal_acc(dset_loaders['test'], netF, netB, netC)
    log_str = '《Source Only》 Task: {}, Test Accuracy on Target = {:.2f}%'.format(args.dset, acc)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str + '\n')


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def train_target(args):
    # plt.figure(num=1, figsize=(10,10))
    # fig2 = plt.figure(num=2)
    # plt.ion()
    dset_loaders = digit_load(args)
    netF = network.DTNBase().cuda()
    # netF = rn.RestNet18()
    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()
    args.modelpath = args.output_dir + '/source_F.pt'
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/source_B.pt'
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/source_C.pt'
    netC.load_state_dict(torch.load(args.modelpath))
    netC.eval()
    epoch = np.array([0])
    acccc = np.array([0])
    for k, v in netC.named_parameters():
        v.requires_grad = False
    param_group = []
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': args.lr2}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': args.lr2}]
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)
    max_iter = args.mini_max_epoch * len(dset_loaders["target"])
    interval_iter = len(dset_loaders["target"])
    iter_num = 0
    while iter_num < max_iter:
        optimizer.zero_grad()
        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_test.next()
        if inputs_test.size(0) == 1:
            continue
        if iter_num % interval_iter == 0 and args.cls_par > 0:
            netF.eval()
            netB.eval()
            mem_label = obtain_label(dset_loaders["target_te"], netF, netB, netC, args)
            mem_label = torch.from_numpy(mem_label).cuda()
            netF.train()
            netB.train()
        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        inputs_test = inputs_test.cuda()
        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test)
        if args.cls_par > 0: #cluster
            pred = mem_label[tar_idx].long()
            classifier_loss = args.cls_par * nn.CrossEntropyLoss()(outputs_test, pred)
        else:
            classifier_loss = torch.tensor(0.0).cuda()

        if args.ent:#local entropy
            softmax_out = nn.Softmax(dim=1)(outputs_test)
            entropy_loss = torch.mean(loss.Entropy(softmax_out))
            if args.gent:#global entropy
                msoftmax = softmax_out.mean(dim=0)
                entropy_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))

            im_loss = entropy_loss * args.ent_par
            classifier_loss += im_loss

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            # if args.drawCM and iter_num != 0:
            #     visualizeCF(args, iter_num / interval_iter, NF=netF, NB=netB, NC=netC)
            print('《SFDA》 all Loss = {:.2f}  im Loss = {:.2f}  pseaudo Loss = {:.2f}'.format(classifier_loss, im_loss, classifier_loss-im_loss))
            netF.eval()
            netB.eval()
            acc, _ = cal_acc(dset_loaders['test'], netF, netB, netC)
            log_str = 'Task: {}, Iter:{}/{}; Test Accuracy on Target Test = {:.2f}%'.format(args.dset, iter_num, max_iter, acc)
            epoch = np.append(epoch, iter_num/interval_iter)
            acccc = np.append(acccc, acc)
            # fig2.clf()
            # plt.plot(epoch, acccc, lw=4, ls='-', c='r', alpha=0.1)
            # plt.title('Target')
            # plt.xlabel('Epoch')
            # plt.ylabel('Accuracy')
            # plt.draw()
            # plt.pause(0.1)
            # plt.ioff()
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')
            netF.train()
            netB.train()

    if args.issave:
        torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))
        torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_" + args.savename + ".pt"))
        torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_" + args.savename + ".pt"))

    return netF, netB, netC


def obtain_label(loader, netF, netB, netC, args, c=None):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == torch.squeeze(all_label)).item() / float(all_label.size()[0])
    # 行向量构成的张量，在特征张量的最后补上一列1
    # 避免小量除以小量
    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    # 2范数进行标准化
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    # 内积矩阵
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
    dd = cdist(all_fea, initc, 'cosine')
    pred_label = dd.argmin(axis=1)
    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)
        all_label = np.squeeze(all_label.float().numpy())
        acc = np.sum(pred_label == all_label) / len(all_fea)
    log_str = 'Last/This Epoch Pseudo-Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str + '\n')
    return pred_label.astype('int')

def plot_embedding_2d(X, y, path=None, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    # Plot colors numbers
    for i in range(len(y)//2):
        plt.text(X[i, 0], X[i, 1], '.', color=plt.cm.Set3(int(y[i])+1), fontdict={'weight': 'bold', 'size': 9})
        plt.text(X[i+len(y)//2, 0], X[i+len(y)//2, 1], 'o', color=plt.cm.Set3(int(y[i+len(y)//2]-args.class_num)+1), fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title)
    if title is not None:
        plt.savefig(path, dpi=300)
    plt.show()

def visualizetsne(args):
    print("Computing t-SNE embedding")
    tsne2d = TSNE(n_components=2, init='pca', random_state=1234)
    dset_loaders = digit_load(args)
    netF = network.DTNBase()
    # netF = rn.RestNet18()
    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck)
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck)
    args.modelpath = args.output_dir + '/source_F.pt'
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/source_B.pt'
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/source_C.pt'
    netC.load_state_dict(torch.load(args.modelpath))
    netF.eval()
    netB.eval()
    netC.eval()
    # max_iter = len(dset_loaders["visualizetsneS"]) + len(dset_loaders["visualizetsneT"])
    max_iter = len(dset_loaders["visualizetsneS"])
    iter_num = 0
    while iter_num < max_iter:
        try:
            inputs_visual, label_visual, _ = iter_test.next()
        except:
            iter_test = iter(dset_loaders["visualizetsneS"])
            inputs_visual, label_visual, _ = iter_test.next()
        inputs_test = inputs_visual
        features_test = netB(netF(inputs_test))
        # features_test = netF(inputs_test)
        outputs = netC(netB(netF(inputs_test)))
        if iter_num == 0:
            XX = features_test.detach().numpy()
            all_output = outputs
            all_label = label_visual
        else:
            XX = np.concatenate((XX, features_test.detach().numpy()), axis=0)
            all_output = torch.cat((all_output, outputs), 0)
            all_label = torch.cat((all_label, label_visual), 0)
        iter_num += 1
    iter_num = 0
    while iter_num < max_iter:
        try:
            inputs_visual, label_visual, _ = iter_test.next()
        except:
            iter_test = iter(dset_loaders["visualizetsneT"])
            inputs_visual, label_visual, _ = iter_test.next()
        inputs_test = inputs_visual
        # features_test = netF(inputs_test)
        features_test = netB(netF(inputs_test))
        outputs = netC(netB(netF(inputs_test)))
        XX = np.concatenate((XX, features_test.detach().numpy()), axis=0)
        all_output = torch.cat((all_output, outputs), 0)
        all_label = torch.cat((all_label, label_visual), 0)
        iter_num += 1

    XX = np.array(XX)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    predict = predict.numpy()
    np.reshape(predict, len(predict))
    all_label = all_label.numpy()
    np.reshape(all_label, len(all_label))
    for i in range(predict.size // 2):
        predict[predict.size // 2 + i] += args.class_num
        all_label[all_label.size // 2 + i] += args.class_num
    Xtsne2d = tsne2d.fit_transform(XX)
    print('Fit ok')
    PATH = "./"+"tsne"+"/"+args.dschool+"/"+args.dset+ ".png"
    PATHH = "./"+"tsne"+"/"+args.dschool+"/"+args.dset+ ".xlsx"

    point_pd = pd.DataFrame(Xtsne2d)
    label_pd = pd.DataFrame(all_label)
    predd_pd = pd.DataFrame(predict)
    writer = pd.ExcelWriter(PATHH)
    point_pd.to_excel(writer, 'point', float_format='%.6f')
    label_pd.to_excel(writer, 'label', float_format='%.6f')
    predd_pd.to_excel(writer, 'predd', float_format='%.6f')
    writer.save()
    writer.close()
    # if args.drawTSNElabel:
    plot_embedding_2d(Xtsne2d[:, 0:2], all_label, path=PATH, title="t-SNE 2D")
    # else:
    #     plot_embedding_2d(Xtsne2d[:, 0:2], predict, path=PATH, title="t-SNE 2D")
    plt.close()

def visualizeCF(args, epoch, NF, NB, NC):
    printl = "Making the Confusion Matrix for " + str(int(epoch)) + " epoch"
    print(printl)
    dset_loaders = digit_load(args)
    NF.eval()
    NB.eval()
    NC.eval()
    max_iter = len(dset_loaders["visualizecf"])
    iter_num = 0
    while iter_num < max_iter:
        try:
            inputs_visual, label_visual, _ = iter_test.next()
        except:
            iter_test = iter(dset_loaders["visualizecf"])
            inputs_visual, label_visual, _ = iter_test.next()
        outputs = NC(NB(NF(inputs_visual.cuda())))
        if iter_num == 0:
            yy = label_visual.detach().numpy()
            all_output = outputs.float()
        else:
            yy = np.concatenate((yy, label_visual.detach().numpy()), axis=0)
            all_output = torch.cat((all_output, outputs.float()), 0)
        iter_num += 1
    yy = np.array(yy)
    # 混淆矩阵
    # 真实标签
    y_actu = yy.flatten()
    # 预测标签
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    y_pred = predict.cpu().detach().numpy()
    cm = ConfusionMatrix(actual_vector=y_actu, predict_vector=y_pred)
    # CFtitle = "ConfusionMatrix of Task " + args.dset + " on " + args.dschool
    cm.plot(cmap=plt.cm.Reds, number_label=True, normalized=True, plot_lib="seaborn", figsizeline=12, figsizecol=10, size=1)
    plt.savefig("./ConfusionMatrix/" + args.dschool + "/" + args.dset + str(int(epoch)) + "epoch.png")
    plt.close()

if __name__ == "__main__":
    warnings.filterwarnings("ignore")  # 忽略警告
    # 命令行读取参数
    parser = argparse.ArgumentParser(description='SFDA')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="GPU device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=100, help="maximum epoch")
    parser.add_argument('--mini_max_epoch', type=int, default=20, help="maximum epoch for train target")
    parser.add_argument('--batch_size', type=int, default=128, help="batch_size")
    parser.add_argument('--worker', type=int, default=0, help="number of workers")
    parser.add_argument('--dset', type=str, default='7252')
    parser.add_argument('--dschool', type=str, default='TT200Hz', choices=['CWRU', 'kaggle', 'MFPT', 'canada', 'cd', 'TorinoA2x', 'TorinoA2y', 'TorinoA2z', 'TorinoA1x', 'TorinoA1y', 'TorinoA1z', 'TT100Hz', 'TT200Hz', 'TT300Hz', 'TT400Hz'],
                        help='school of dataset')
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate for train source")
    parser.add_argument('--lr2', type=float, default=0.0001, help="learning rate for train target")
    parser.add_argument('--seed', type=int, default=2022, help="random seed")
    parser.add_argument('--cls_par', type=float, default=0.1, help="the weight of cluster loss")
    parser.add_argument('--ent_par', type=float, default=1.0, help="the weight of entropy loss")
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True, help="information maximum")
    parser.add_argument('--bottleneck', type=int, default=1024)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.2)
    parser.add_argument('--output', type=str, default='ckps_digits')
    parser.add_argument('--issave', type=bool, default=False)
    parser.add_argument('--drawCM', type=bool, default=False)
    parser.add_argument('--drawTSNE', type=bool, default=True)
    parser.add_argument('--drawTSNElabel', type=bool, default=True, help="Tsne with ground truth or predict")
    args = parser.parse_args()
    args.class_num = 10
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.cuda.manual_seed(SEED)

    args.output_dir = osp.join(args.output, 'seed' + str(args.seed), args.dset)
    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if not osp.exists(osp.join(args.output_dir + '/source_F.pt')):
        args.out_file = open(osp.join(args.output_dir, 'log_src.txt'), 'w')
        args.out_file.write(print_args(args) + '\n')
        args.out_file.flush()
        train_source(args)
        target_test(args)

    args.savename = 'par_' + str(args.cls_par)
    args.out_file = open(osp.join(args.output_dir, 'log_tar_' + args.savename + '.txt'), 'w')
    args.out_file.write(print_args(args) + '\n')
    args.out_file.flush()
    train_target(args)
    # if args.drawTSNE:
    #     visualizetsne(args)
    print("Finish !")