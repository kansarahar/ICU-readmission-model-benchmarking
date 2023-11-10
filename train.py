import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, f1_score

import torch
import torch.nn as nn

from load_data import get_data_loader, DatasetType
from modules.modules import Network, ODE_RNN

if __name__ == '__main__':

    # args
    parser = argparse.ArgumentParser(description='Train models to predict ICU readmission in patients')
    parser.add_argument('--data_path', dest='data_path', type=str, default='./data/preprocessed/data_arrays.npz', help='path to data_arrays.npz file')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='batch size (default: 1)')
    parser.add_argument('--epochs', dest='epochs', type=int, default=1, help='number of epochs (default: 1)')
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--model_type', dest='model_type', type=str, choices=['ode_rnn'], default='ode_rnn', help='type of model you want to train (default: ode_rnn)')
    parser.add_argument('--save_destination', dest='save_dest', type=str, default='./trained_models')
    args = parser.parse_args()

    # data
    dir_name = os.path.dirname(os.path.abspath(__file__))
    data_arrays = np.load(os.path.join(dir_name, args.data_path), allow_pickle=True)

    # cuda
    use_cuda = torch.cuda.is_available()
    if (not use_cuda): print('CUDA is unavailable - using CPU')
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    # model
    save_path = os.path.abspath(os.path.join(dir_name, args.save_dest, args.model_type + '.pt'))
    model_map = {
        'ode_rnn': ODE_RNN
    }
    if args.model_type not in model_map:
        sys.exit('Invalid model type - run "python train.py -h" for more info')
    model = model_map[args.model_type](len(data_arrays['static_vars']), data_arrays['dp'].max() + 1, data_arrays['cp'].max() + 1).to(device)
    if (os.path.isfile(save_path)):
        print('Loading Existing Model:', args.model_type)
        model.load_state_dict(torch.load(save_path))
    else:
        print('Training New Model:', args.model_type)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(0, args.epochs):

        train_loader, train_pos_weight = get_data_loader(os.path.join(dir_name, args.data_path), DatasetType.TRAIN, args.batch_size)
        validation_loader, _ = get_data_loader(os.path.join(dir_name, args.data_path), DatasetType.VALIDATE, args.batch_size)
        loss_function = nn.BCEWithLogitsLoss(pos_weight=train_pos_weight)
        training_loss = 0
        validation_loss = 0

        # train
        print('Training %s model (Epoch %s)' % (args.model_type, epoch))
        model.train()
        for i, (static, dp, cp, dp_times, cp_times, label) in enumerate(tqdm(train_loader)):
            static: torch.tensor = static.to(device)
            dp: torch.tensor = dp.to(device)
            cp: torch.tensor = cp.to(device)
            dp_times: torch.tensor = dp_times.to(device)
            cp_times: torch.tensor = cp_times.to(device)
            label: torch.tensor = label.to(device)

            optimizer.zero_grad()

            model_out = model(static, dp, cp, dp_times, cp_times).flatten()
            loss = loss_function(model_out, label)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

        # validate
        print('Validating %s model (Epoch %s)' % (args.model_type, epoch))
        model.eval()
        with torch.no_grad():
            preds = np.array([])
            labels = np.array([])
            for i, (static, dp, cp, dp_times, cp_times, label) in enumerate(tqdm(validation_loader)):
                static: torch.tensor = static.to(device)
                dp = dp.to(device)
                cp = cp.to(device)
                dp_times = dp_times.to(device)
                cp_times = cp_times.to(device)
                label = label.to(device)

                model_out = model(static, dp, cp, dp_times, cp_times).flatten()
                loss = loss_function(model_out, label)
                pred = torch.sigmoid(model_out).cpu().numpy()
                y = label.cpu().numpy()
                preds = np.concatenate((preds, pred))
                labels = np.concatenate((labels, y))
                validation_loss += loss.item()

        preds[preds <= 0.5] = 0
        preds[preds > 0.5] = 1
        accuracy = accuracy_score(labels, preds)
        precision = average_precision_score(labels, preds)
        auroc = roc_auc_score(labels, preds)
        f1 = f1_score(labels, preds)

        print('\nValidation Results (Epoch %s)' % epoch)
        print('Accuracy Score: %s' % accuracy)
        print('Average Precision Score: %s' % precision)
        print('ROC AUC Score: %s' % auroc)
        print('F1 Score: %s' % f1)
        print('Training Loss: %s' % training_loss)
        print('Validation Loss: %s' % validation_loss)

        training_loss = 0
        validation_loss = 0

        os.makedirs(args.save_dest, exist_ok=True)
        print('Saving model to %s ...' % save_path)
        torch.save(model.state_dict(), save_path)
        print('Saved model')
