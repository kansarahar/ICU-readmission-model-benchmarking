import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Type
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, f1_score

import torch
import torch.nn as nn

from load_data import get_data_loader, DatasetType
from modules.modules import Network, ODE_RNN, RNN_Exp_Decay, RNN_Concat_Time_Delta

if __name__ == '__main__':

    # args
    parser = argparse.ArgumentParser(description='Test trained models to predict ICU readmission in patients')
    parser.add_argument('--data_path', dest='data_path', type=str, default='./data/preprocessed/data_arrays.npz', help='path to data_arrays.npz file')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='batch size (default: 1)')
    parser.add_argument('--model_type', dest='model_type', type=str, choices=['ode_rnn', 'rnn_exp_decay', 'rnn_concat_time_delta'], default='ode_rnn', help='type of model you want to test (default: ode_rnn)')
    parser.add_argument('--save_destination', dest='save_dest', type=str, default='./trained_models')
    args = parser.parse_args()

    # data
    dir_name = os.path.dirname(os.path.abspath(__file__))
    data_arrays = np.load(os.path.join(dir_name, args.data_path), allow_pickle=True)

    # cuda
    use_cuda = torch.cuda.is_available()
    if (not use_cuda): print('Warn: CUDA is unavailable - using CPU')
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    # model
    save_path = os.path.abspath(os.path.join(dir_name, args.save_dest, args.model_type + '.pt'))
    model_map = {
        'ode_rnn': ODE_RNN,
        'rnn_exp_decay': RNN_Exp_Decay,
        'rnn_concat_time_delta': RNN_Concat_Time_Delta,
    }
    if args.model_type not in model_map:
        sys.exit('Invalid model type - run "python test.py -h" for more info')
    model_type: Type[Network] = model_map[args.model_type]
    model = model_type(len(data_arrays['static_vars']), data_arrays['dp'].max() + 1, data_arrays['cp'].max() + 1, dropout_probability=0.2, is_bidirectional=True, device=device)
    model = model.to(device)
    if (os.path.isfile(save_path)):
        print('Loading Existing Model:', args.model_type)
        model.load_state_dict(torch.load(save_path))
    else:
        sys.exit('Could not find model: %s' % save_path)


    test_loader, test_pos_weight = get_data_loader(os.path.join(dir_name, args.data_path), DatasetType.TEST, args.batch_size)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=test_pos_weight)
    test_loss = 0

    # test
    print('Testing %s model' % args.model_type)
    model.eval()
    with torch.no_grad():
        preds = np.array([])
        labels = np.array([])
        for i, (static, dp, cp, dp_times, cp_times, label) in enumerate(tqdm(test_loader)):
            static: torch.tensor = static.to(device)
            dp = dp.to(device)
            cp = cp.to(device)
            dp_times = dp_times.to(device)
            cp_times = cp_times.to(device)
            label = label.to(device)

            model_out = model(static, dp, cp, dp_times, cp_times).flatten()
            loss = loss_function(model_out, label)
            test_loss += loss.item()
            pred = torch.sigmoid(model_out).cpu().numpy()
            y = label.cpu().numpy()
            preds = np.concatenate((preds, pred))
            labels = np.concatenate((labels, y))
            test_loss += loss.item()

    preds[preds <= 0.5] = 0
    preds[preds > 0.5] = 1
    accuracy = accuracy_score(labels, preds)
    precision = average_precision_score(labels, preds)
    auroc = roc_auc_score(labels, preds)
    f1 = f1_score(labels, preds)

    print('\nTest Results')
    print('Accuracy Score: %s' % accuracy)
    print('Average Precision Score: %s' % precision)
    print('ROC AUC Score: %s' % auroc)
    print('F1 Score: %s' % f1)
    print('Test Loss: %s' % test_loss)
