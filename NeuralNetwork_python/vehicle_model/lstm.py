import numpy as np
import time
import threading
import os
import logging
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.signal import firwin
from torch.utils.data import Dataset, DataLoader, random_split


def process_targets(data_length, early_rul=None):
    """
    Takes datalength and earlyrul as input and
    creates target rul.
    """
    if early_rul == None:
        return np.arange(data_length - 1, -1, -1)
    else:
        early_rul_duration = data_length - early_rul
        if early_rul_duration <= 0:
            return np.arange(data_length - 1, -1, -1)
        else:
            return np.append(
                early_rul * np.ones(shape=(early_rul_duration,)),
                np.arange(early_rul - 1, -1, -1),
            )


def process_input_data_with_targets(
    input_data, target_data=None, window_length=1, shift=1
):
    """Depending on values of window_length and shift, this function generates batchs of data and targets
    from input_data and target_data.

    Number of batches = np.floor((len(input_data) - window_length)/shift) + 1

    **We don't check input dimensions uisng exception handling. So readers should be careful while using these
    functions. If input data are not of desired dimension, either error occurs or something undesirable is
    produced as output.**

    Arguments:
        input_data: input data to function (Must be 2 dimensional)
        target_data: input rul values (Must be 1D array)s
        window_length: window length of data
        shift: Distance by which the window moves for next batch. This is closely related to overlap
               between data. For example, if window length is 30 and shift is 1, there is an overlap of
               29 data points between two consecutive batches.

    """
    num_batches = np.int(np.floor((len(input_data) - window_length) / shift)) + 1
    num_features = input_data.shape[1]
    output_data = np.repeat(
        np.nan, repeats=num_batches * window_length * num_features
    ).reshape(num_batches, window_length, num_features)
    if target_data is None:
        for batch in range(num_batches):
            output_data[batch, :, :] = input_data[
                (0 + shift * batch) : (0 + shift * batch + window_length), :
            ]
        return output_data
    else:
        output_targets = np.repeat(np.nan, repeats=num_batches)
        for batch in range(num_batches):
            output_data[batch, :, :] = input_data[
                (0 + shift * batch) : (0 + shift * batch + window_length), :
            ]
            output_targets[batch] = target_data[(shift * batch + (window_length - 1))]
        return output_data, output_targets


def process_test_data(
    test_data_for_an_engine, window_length, shift, num_test_windows=1
):
    """This function takes test data for an engine as first input. The next two inputs
    window_length and shift are same as other functins.

    Finally it takes num_test_windows as the last input. num_test_windows sets how many examplles we
    want from test data (from last). By default it extracts only the last example.

    The function return last examples and number of last examples (a scaler) as output.
    We need the second output later. If we are extracting more than 1 last examples, we have to
    average their prediction results. The second scaler halps us do just that.
    """
    max_num_test_batches = (
        np.int(np.floor((len(test_data_for_an_engine) - window_length) / shift)) + 1
    )
    if max_num_test_batches < num_test_windows:
        required_len = (max_num_test_batches - 1) * shift + window_length
        batched_test_data_for_an_engine = process_input_data_with_targets(
            test_data_for_an_engine[-required_len:, :],
            target_data=None,
            window_length=window_length,
            shift=shift,
        )
        return batched_test_data_for_an_engine, max_num_test_batches
    else:
        required_len = (num_test_windows - 1) * shift + window_length
        batched_test_data_for_an_engine = process_input_data_with_targets(
            test_data_for_an_engine[-required_len:, :],
            target_data=None,
            window_length=window_length,
            shift=shift,
        )
        return batched_test_data_for_an_engine, num_test_windows
