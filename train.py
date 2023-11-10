import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn

from load_data import get_data_loader, DatasetType
from modules.modules import Network, ODE_RNN

if __name__ == '__main__':

    # args
    parser = argparse.ArgumentParser(description='Train models to predict ICU readmission in patients')
    parser.add_argument('--data_path', '-d', dest='data_path', type=str, default='./data/preprocessed/data_arrays.npz', help='path to data_arrays.npz file')
    parser.add_argument('--batch_size', '-b', dest='batch_size', type=int, default=32, help='batch size (default: 1)')
    parser.add_argument('--epochs', '-e', dest='epochs', type=int, default=1, help='number of epochs (default: 1)')
    parser.add_argument('--learning_rate', '-lr', dest='learning_rate', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--model_type', '-m', dest='model_type', type=str, choices=['ode_rnn'], default='ode_rnn', help='type of model you want to train (default: ode_rnn)')
    parser.add_argument('--save_destination', '-s', dest='save_dest', type=str, default='./trained_models')
    args = parser.parse_args()

    # data
    dir_name = os.path.dirname(os.path.abspath(__file__))
    data_arrays = np.load(os.path.join(dir_name, args.data_path), allow_pickle=True)

    # cuda
    use_cuda = torch.cuda.is_available()
    if (not use_cuda): print('CUDA is unavailable - using CPU')
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    # model
    model_map = {
        'ode_rnn': ODE_RNN
    }
    if args.model_type not in model_map:
        sys.exit('Invalid model type - run "python train.py -h" for more info')
    model = model_map[args.model_type](len(data_arrays['static_vars']), data_arrays['dp'].max() + 1, data_arrays['cp'].max() + 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # train
    for epoch in range(0, args.epochs):

        train_loader, train_pos_weight = get_data_loader(os.path.join(dir_name, args.data_path), DatasetType.TRAIN, args.batch_size)
        validation_loader, _ = get_data_loader(os.path.join(dir_name, args.data_path), DatasetType.VALIDATE, args.batch_size)
        loss_function = nn.BCEWithLogitsLoss(pos_weight=train_pos_weight)
        epoch_loss = 0
        total_loss = 0
        for i, (static, dp, cp, dp_times, cp_times, label) in enumerate(tqdm(train_loader)):
            static: torch.tensor = static.to(device)
            dp: torch.tensor = dp.to(device)
            cp: torch.tensor = cp.to(device)
            dp_times: torch.tensor = dp_times.to(device)
            cp_times: torch.tensor = cp_times.to(device)
            label: torch.tensor = label.to(device)

            optimizer.zero_grad()

            pred = model(static, dp, cp, dp_times, cp_times)
            loss = loss_function(pred, label.unsqueeze(1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            total_loss += loss.item()
            if i % 10 == 0:
                print('\nloss:', total_loss)
                total_loss = 0
        
        os.makedirs(args.save_dest, exist_ok=True)
        save_path = os.path.abspath(os.path.join(args.save_dest, args.model_type + '.pt'))
        print('Saving model to %s ...' % save_path)
        torch.save(model.state_dict(), save_path)
        print('Saved model')
