# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

#!/usr/bin/env python
import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging
import random
from generate_cifar10 import generate_cifar10
from generate_cifar100 import generate_cifar100
from generate_mnist import generate_mnist

from flcore.servers.serverpfakd import PFAKD
from flcore.trainmodel.models import *
from utils.result_utils import average_data
from utils.mem_utils import MemReporter

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
torch.manual_seed(42)
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# hyper-params for Text tasks
vocab_size = 98635   #98635 for AG_News and 399198 for Sogou_News
max_len=200
emb_dim=32

def run(args):
    setup_seed(42)
    time_list = []
    reporter = MemReporter()
    model_str = args.model

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()
        if model_str == "CNN":  # non-convex
            if "fmnist" in args.dataset:
                args.model = CNN_FMNIST(num_classes=args.num_classes).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = CNN(num_classes=args.num_classes).to(args.device)
            elif "Cifar100" in args.dataset:
                args.model = CNN(num_classes=args.num_classes).to(args.device)
        elif model_str == "ResNet10":
            args.model = ResNet(BasicBlock, [1, 1, 1, 1], args.num_classes).to(args.device)
        elif model_str == "ResNet18":
            args.model = ResNet(BasicBlock, [2, 2, 2, 2], args.num_classes).to(args.device)
        elif model_str == "ResNet34":
            args.model = ResNet(BasicBlock, [3, 4, 6, 3], args.num_classes).to(args.device)

        else:
            raise NotImplementedError

        print(args.model)

        # select algorithm

        if args.algorithm == "PFAKD":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head, args.algorithm)
            server = FedPer(args, i)
        else:
            raise NotImplementedError

        server.train()

        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    

    # Global average
    # average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)

    print("All done!")

    reporter.report()


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--niid', type=bool, default=True)  # 是否niid
    parser.add_argument('--partition', type=str, default="dir")  # 数据集划分方式 狄雷克雷
    parser.add_argument('--distill', type=bool, default=True)  # 是否蒸馏         ########################
    parser.add_argument('--B', type=float, default=1)  # 蒸馏损失权重           ###################
    parser.add_argument('-go', "--goal", type=str, default="test", 
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0") # 显卡编号                ##############
    parser.add_argument('-data', "--dataset", type=str, default="Cifar100")  # 数据集            #############
    parser.add_argument('-nb', "--num_classes", type=int, default=100)   # 分类                 ###########
    parser.add_argument('-m', "--model", type=str, default="ResNet18")    # 模型      ###############
    parser.add_argument('-lbs', "--batch_size", type=int, default=128) # 批量大小
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.01,  # 学习率   #############
                        help="Local learning rate")
    parser.add_argument('-wd', "--weight_decay", type=bool, default=True)  # 权重衰减和动量     #############
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=True)  # 学习率衰减      #################
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.9)  # 学习率衰减系数  ##################
    parser.add_argument('-gr', "--global_rounds", type=int, default=40) # 全局轮数    #############
    parser.add_argument('-ls', "--local_epochs", type=int, default=10,       # 本地轮数   ###############FedRep的主体更新次数
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="PFAKD")   # 使用算法    ##############
    # pFedMe / PerAvg / FedProx / FedAMP / FedPHP / GPFL/FedPAC
    parser.add_argument('-lamda_pac', "--lamda_pac", type=float, default=0.001,   #############FedPAC
                        help="Regularization weight-pac")
    parser.add_argument('-lam', "--lamda", type=float, default=0.0001,   #############GPFL
                        help="Regularization weight")
    parser.add_argument('-mu', "--mu", type=float, default=1)     ################ GPFL
    # Ditto / FedRep
    parser.add_argument('-pls', "--plocal_epochs", type=int, default=10)   ##################FedRep  head的更新次数


    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)

    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local epochs: {}".format(args.local_epochs))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Local learing rate decay: {}".format(args.learning_rate_decay))
    if args.learning_rate_decay:
        print("Local learing rate decay gamma: {}".format(args.learning_rate_decay_gamma))
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    print("Clients randomly join: {}".format(args.random_join_ratio))
    print("Client drop rate: {}".format(args.client_drop_rate))
    print("Client select regarding time: {}".format(args.time_select))
    if args.time_select:
        print("Time threthold: {}".format(args.time_threthold))
    print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
    print("Number of classes: {}".format(args.num_classes))
    print("Backbone: {}".format(args.model))
    print("Using device: {}".format(args.device))
    print("Using DP: {}".format(args.privacy))
    if args.privacy:
        print("Sigma for DP: {}".format(args.dp_sigma))
    print("Auto break: {}".format(args.auto_break))
    if not args.auto_break:
        print("Global rounds: {}".format(args.global_rounds))
    if args.device == "cuda":
        print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print("DLG attack: {}".format(args.dlg_eval))
    if args.dlg_eval:
        print("DLG attack round gap: {}".format(args.dlg_gap))
    print("Total number of new clients: {}".format(args.num_new_clients))
    print("Fine tuning epoches on new clients: {}".format(args.fine_tuning_epoch_new))
    print("=" * 50)
