import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

if __name__ == '__main__':

    ################################# args ################################

    dir_name = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='A 1D function approximator playground built with pytorch')

    # training params
    parser.add_argument('--batch-size', '-b', dest='batch_size', type=int, default=1, help='batch size (default: 1)')
    parser.add_argument('--epochs', '-e', dest='epochs', type=int, default=1, help='number of epochs (default: 1)')
    parser.add_argument('--learning-rate', '-lr', dest='lr', type=float, default=0.001, help='learning rate (default: 0.001)')

    args = parser.parse_args()
