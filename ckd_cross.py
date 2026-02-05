from datetime import datetime
import torch
import numpy as np
import argparse

from tqdm import tqdm

from model.Trainer import backbone_network
import h5py
import os
import time
import pandas
import random
from my_utils.dataloader_student import MyDataSet_train, MyDataSet_test
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from aug_nor import augment_eeg_weak, augment_eeg_strong, augment_face_weak, augment_face_strong
from aug import standardize_data, augment_data

os.environ["NUMEXPR_MAX_THREADS"] = "32"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--num_epoch', type=int, default=400)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_class', type=int, default=4)
parser.add_argument('--dim_eeg', type=int, default=60)
parser.add_argument('--dim_channel', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--max_grad_norm', type=float, default=5.0)
parser.add_argument('--dropout', default=0.1, type=float)
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--cpu', action='store_true')
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--seed', default=1024, type=int)
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--in_channels', type=int, default=4)
parser.add_argument('--out_channels', type=int, default=4)
parser.add_argument('--kernel_size', type=int, default=3)
parser.add_argument('--padding', type=int, default=1)
# bi-lstm
parser.add_argument('--dim_face', type=int, default=60)
parser.add_argument('--bilstm_hidden', type=int, default=64)

args = parser.parse_args()
opt = vars(args)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

init_time = time.time()

de_path = '/home/user/MyDisk/srs/DE_allbands.mat'
face_path = '/home/user/MyDisk/srs/AUG/code/data/Face_AV.mat'

dataset = h5py.File(de_path)
DE_features = dataset["DE_features"]
theta = torch.from_numpy(np.array(DE_features["theta"])).float()
alpha = torch.from_numpy(np.array(DE_features["alpha"])).float()
beta = torch.from_numpy(np.array(DE_features["beta"])).float()
gamma = torch.from_numpy(np.array(DE_features["gamma"])).float()

combined_data = torch.stack([theta, alpha, beta, gamma], dim=1)
data_numpy = np.array(combined_data).transpose(3, 1, 2, 0)

xx_e = data_numpy[0:880, 0:4, ::]
EEG_data = torch.tensor(xx_e)
labels = np.array(dataset["AV_labels"]).reshape(-1)
labels = labels[0:880]

dataset = h5py.File(face_path)
one_data1 = dataset["face"]
xx_f = np.array(one_data1).transpose(3, 0, 1, 2)
Face_data = torch.tensor(xx_f)

labels = torch.tensor(labels).long()
EEG_data = EEG_data.float()
Face_data = Face_data.float()
###增大样本
labels = torch.tensor(labels)
EEG_data, Face_data, labels = augment_data(EEG_data, Face_data, labels, times=2)
labels = labels.numpy()
##############data_normalization:0~1
EEG_data, Face_data = standardize_data(EEG_data, Face_data)

#eeg_weak = augment_eeg_weak(EEG_data)
#eeg_strong = augment_eeg_strong(EEG_data)
#face_weak = augment_face_weak(Face_data)
#face_strong = augment_face_strong(Face_data)
#eeg_augmented = torch.stack([eeg_weak, eeg_strong], dim=1)
#face_augmented = torch.stack([face_weak, face_strong], dim=1)

df = pandas.DataFrame(columns=['time', 'Fold', 'beEpoch', 'epo_beAcc', 'epo_beAcc0', 'epo_beAcc1', 'epo_beAcc2', 'epo_beAcc3'])
# df = pandas.DataFrame(columns=['time', 'Fold', 'beEpoch', 'epo_beAcc', 'epo_beAcc0', 'epo_beAcc1'])
# root_path = 'results/DEAP/AV'
path = 'results/DEAP/AV'

local_time = time.localtime()[0:3]
# csv_name = 'cross_1*1* 1_4*1' + '_{:02d}_{:02d}{:02d}'.format(local_time[0], local_time[1], local_time[2]) + '.csv'
csv_name = 'bilstm' + '_{:02d}_{:02d}{:02d}'.format(local_time[0], local_time[1], local_time[2]) + '.csv'
df.to_csv(os.path.join(path, csv_name), index=False)

global_start_time = time.time()

K = 5
# kf = KFold(n_splits=K, shuffle=True, random_state=42)
kf = StratifiedKFold(n_splits= K, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(kf.split(EEG_data, labels)):
    print(f"\n=== Fold {fold + 1}/{K} ===")

    #data_x1_train = eeg_augmented[train_idx]
    #data_x2_train = face_augmented[train_idx]
    data_x1_train, data_x1_test = EEG_data[train_idx], EEG_data[test_idx]
    data_x2_train, data_x2_test = Face_data[train_idx], Face_data[test_idx]
    lab_train = labels[train_idx]
    #data_x1_test = EEG_data[test_idx]
    #data_x2_test = Face_data[test_idx]
    lab_test = labels[test_idx]

    opt['te_batch_size'] = max(1, lab_test.shape[0] // 2)
    opt['tr_batch_size'] = max(1, lab_train.shape[0] // 8)

    print(f"Train data shape: EEG {data_x1_train.shape}, Face {data_x2_train.shape}")
    print(f"Test data shape: EEG {data_x1_test.shape}, Face {data_x2_test.shape}")

    model = backbone_network(opt)
    if args.cuda:
        model.cuda()

    best_acc = 0
    best_epoch = 0

    train_dataset = MyDataSet_train(data_x1_train, data_x2_train, lab_train)
    train_loader = DataLoader(train_dataset, batch_size=opt['tr_batch_size'], shuffle=True)

    for epoch in range(1, opt['num_epoch'] + 1):
        model.training = True
        train_loss = 0
        train_acc = 0
        train_acc_per_class = [0, 0, 0, 0]
        # train_acc_per_class = [0, 0]
        count = 0

        train_loader_tqdm = tqdm(enumerate(train_loader),
                                 total=len(train_loader),
                                 desc=f"Epoch {epoch}/{opt['num_epoch']}",
                                 bar_format="{l_bar}{bar:20}{r_bar}",
                                 dynamic_ncols=True)


        for tr_idx, (train_x1, train_x2, train_y) in train_loader_tqdm:
            if args.cuda:
                train_x1 = train_x1.cuda()
                train_x2 = train_x2.cuda()
                train_y = train_y.cuda()

            log, loss = model.train_oneStep(train_x1, train_x2, train_y)

            _, pred_class = torch.max(log.cpu(), dim=1)
            train_y_cpu = train_y.cpu()

            unique_labels = np.unique(train_y_cpu)
            if len(unique_labels) < 2:
                # print(f"Train_Batch {tr_idx + 1} only contains classes {unique_labels}, skipping metrics.")
                continue

            accuracy = accuracy_score(train_y_cpu, pred_class)
            count += 1

            batch_class_acc = []
            for label in [0, 1, 2, 3]:
            # for label in [0, 1]:
                label_indices = (train_y_cpu == label)
                if label_indices.sum() > 0:
                    class_acc = accuracy_score(train_y_cpu[label_indices], pred_class[label_indices])
                    train_acc_per_class[label] += class_acc
                    batch_class_acc.append(f"{label}:{class_acc:.2f}")

            train_loss += loss.item()
            train_acc += accuracy

            avg_loss = train_loss / count if count > 0 else 0
            avg_acc = train_acc / count if count > 0 else 0
            class_acc_str = " | ".join(batch_class_acc)

            train_loader_tqdm.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                'acc': f'{accuracy:.4f}',
                'avg_acc': f'{avg_acc:.4f}',
                'classes': class_acc_str
            })

        if count > 0:
            train_loss /= count
            train_acc /= count
            train_acc_per_class = [acc / count for acc in train_acc_per_class]

        # print(f"Fold {fold+1}, Epoch {epoch}: Train Acc: {train_acc:.4f}, Loss: {train_loss:.4f}")

        # 测试
        model.training = False
        test_dataset = MyDataSet_test(data_x1_test, data_x2_test, lab_test)
        test_loader = DataLoader(test_dataset, batch_size=opt['te_batch_size'], shuffle=False)

        test_acc = 0
        test_acc_per_class = [0, 0, 0, 0]
        # test_acc_per_class = [0, 0,]
        test_count = 0

        test_loader_tqdm = tqdm(enumerate(test_loader),
                                total=len(test_loader),
                                desc=f"Testing",
                                bar_format="{l_bar}{bar:20}{r_bar}",
                                dynamic_ncols=True)

        with torch.no_grad():
            for te_idx, (test_x1, test_x2, test_y) in test_loader_tqdm:
                if args.cuda:
                    test_x1 = test_x1.cuda()
                    test_x2 = test_x2.cuda()
                    test_y = test_y.cuda()

                predicts = model.predict(test_x1, test_x2, test_y)
                _, pred_class = torch.max(predicts.cpu(), dim=1)
                test_y_cpu = test_y.cpu()

                unique_labels = np.unique(test_y_cpu)
                if len(unique_labels) < 4:
                # if len(unique_labels) < 2:   
                    print(f"Test_Batch {te_idx + 1} only contains classes {unique_labels}, skipping metrics.")
                    continue

                accuracy = accuracy_score(test_y_cpu, pred_class)
                test_count += 1

                batch_class_acc = []
                # for label in [0, 1]:
                for label in [0, 1, 2, 3]:
                    label_indices = (test_y_cpu == label)
                    if label_indices.sum() > 0:
                        class_acc = accuracy_score(test_y_cpu[label_indices], pred_class[label_indices])
                        test_acc_per_class[label] += class_acc
                        batch_class_acc.append(f"{label}:{class_acc:.2f}")

                test_acc += accuracy

                avg_test_acc = test_acc / test_count if test_count > 0 else 0
                test_loader_tqdm.set_postfix({
                    'acc': f'{accuracy:.4f}',
                    'avg_acc': f'{avg_test_acc:.4f}',
                    'classes': " | ".join(batch_class_acc)
                })

        if test_count > 0:
            test_acc /= test_count
            test_acc_per_class = [acc / test_count for acc in test_acc_per_class]

        print(f"\nEpoch {epoch} Summary:")
        print(f"Train => Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Test  => Acc: {test_acc:.4f}")
        print("Class Accuracies:")
        print(f"  Train: [{' | '.join([f'{acc:.4f}' for acc in train_acc_per_class])}]")
        print(f"  Test:  [{' | '.join([f'{acc:.4f}' for acc in test_acc_per_class])}]")

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            be_acc0, be_acc1, be_acc2, be_acc3 = test_acc_per_class
            # be_acc0, be_acc1 = test_acc_per_class

    print(f"Fold {fold+1} Best Epoch {best_epoch}: Test Acc {best_acc:.4f}")

    times = "%s" % datetime.now()
    # col = [times, fold + 1, best_epoch, best_acc, be_acc0, be_acc1]
    col = [times, fold + 1, best_epoch, best_acc, be_acc0, be_acc1, be_acc2, be_acc3]
    res = pandas.DataFrame([col])
    res.to_csv(os.path.join(path, csv_name), mode='a', header=False, index=False)

duration = time.time() - global_start_time
print(f"\nTotal Duration: {duration:.2f} seconds")
