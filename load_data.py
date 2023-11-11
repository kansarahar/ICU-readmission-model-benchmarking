import os
import sys
from enum import Enum
import numpy as np
import torch
import torch.utils.data as utils


DatasetType = Enum('DatasetType', ['TRAIN', 'VALIDATE', 'TEST'])

def get_data_loader(path_to_data: str, type: DatasetType, batch_size, shuffle=True, num_workers=1):

    data = np.load(path_to_data, allow_pickle=True)

    static = data['static'].astype('float32')
    label = data['label'].astype('float32')
    dp = data['dp'].astype('int64')
    cp = data['cp'].astype('int64')
    dp_times = data['dp_times'].astype('float32')
    cp_times = data['cp_times'].astype('float32')
    train_ids = data['train_ids']
    validate_ids = data['validate_ids']
    test_ids = data['test_ids']

    if (type == DatasetType.TRAIN):
        ids = train_ids
    elif (type == DatasetType.VALIDATE):
        ids = validate_ids
    else:
        ids = test_ids

    static = static[ids, :]
    label = label[ids]
    dp = dp[ids, :]
    cp = cp[ids, :]
    dp_times = dp_times[ids, :]
    cp_times = cp_times[ids, :]

    dataset = utils.TensorDataset(
        torch.from_numpy(static),
        torch.from_numpy(dp),
        torch.from_numpy(cp),
        torch.from_numpy(dp_times),
        torch.from_numpy(cp_times),
        torch.from_numpy(label)
    )

    data_loader = utils.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True
    )

    pos_weight = torch.tensor((len(label) - np.sum(label)) / np.sum(label))
    
    return data_loader, pos_weight