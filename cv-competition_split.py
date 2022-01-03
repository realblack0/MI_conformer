#!/usr/bin/env python
# coding: utf-8


# 1. Data load
#     1. load
#     2. s to cnt
#     3. label index (1\~4 -> 0\~3)
#     4. pos index (start from 0)

# In[1]:


from scipy.io import loadmat
import mne
import numpy as np
from copy import deepcopy
from functools import wraps


def verbose_func_name(fn):
    @wraps(fn)
    def inner(*args, **kwargs):
        if ("verbose" in kwargs.keys()) and kwargs["verbose"]:
            print("\n" + fn.__name__)
        return fn(*args, **kwargs)

    return inner


# @verbose_func_name
# def load_gdf2mat(subject, train=True, data_dir=".", overflowdetection=True, verbose=False):
#     # Configuration
#     if train:
#         filename = f"A0{subject}T_gdf"
#     else:
#         filename = f"A0{subject}E_gdf"
#     base = data_dir

#     # Load mat files
#     data_path  =  base + '/gdf2mat/' + filename + '.mat'
#     label_path =  base + '/true_labels/' + filename[:4] + '.mat'

#     if not overflowdetection:
#         filename  = filename + "_overflowdetection_off"
#         data_path = base + '/gdf2mat_overflowdetection_off/' + filename + '.mat'

#     session_data = loadmat(data_path, squeeze_me=False)
#     label_data   = loadmat(label_path, squeeze_me=False)

#     # Parse data
#     s = session_data["s"] # signal
#     h = session_data["h"] # header
#     labels = label_data["classlabel"] # true label

#     h_names = h[0][0].dtype.names # header is structured array
#     origin_filename = h["FileName"][0,0][0]
#     train_labels = h["Classlabel"][0][0] # For Evaluation data, it is filled with NaN.
#     artifacts = h["ArtifactSelection"][0,0]

#     events = h['EVENT'][0,0][0,0] # void
#     typ = events['TYP']
#     pos = events['POS']
#     fs  = events['SampleRate'].squeeze()
#     dur = events['DUR']

#     # http://www.bbci.de/competition/iv/desc_2a.pdf
#     typ2desc = {276:'Idling EEG (eyes open)',
#                 277:'Idling EEG (eyes closed)',
#                 768:'Start of a trial',
#                 769:'Cue onset left (class 1)',
#                 770:'Cue onset right (class 2)',
#                 771:'Cue onset foot (class 3)',
#                 772:'Cue onset tongue (class 4)',
#                 783:'Cue unknown',
#                 1024:'Eye movements',
#                 32766:'Start of a new run'}

#     # 출처... 아마... brain decode...
#     ch_names = ['Fz',  'FC3', 'FC1', 'FCz', 'FC2',
#                  'FC4', 'C5',  'C3',  'C1',  'Cz',
#                  'C2',  'C4',  'C6',  'CP3', 'CP1',
#                  'CPz', 'CP2', 'CP4', 'P1',  'Pz',
#                  'P2',  'POz', 'EOG-left', 'EOG-central', 'EOG-right']

#     assert filename[:4] == origin_filename[:4]
#     if verbose:
#         print("- filename:", filename)
#         print("- load data from:", data_path)
#         print('\t- original fileanme:', origin_filename)
#         print("- load label from:", label_path)
#         print("- shape of s", s.shape) # (time, 25 channels),
#         print("- shape of labels", labels.shape) # (288 trials)

#     data =  {"s":s, "h":h, "labels":labels, "filename":filename, "artifacts":artifacts, "typ":typ, "pos":pos, "fs":fs, "dur":dur, "typ2desc":typ2desc, "ch_names":ch_names}
#     return data


@verbose_func_name
def load_gdf2mat_feat_mne(
    subject, train=True, data_dir=".", overflowdetection=True, verbose=False
):
    # Configuration
    if train:
        filename = f"A0{subject}T_gdf"
    else:
        filename = f"A0{subject}E_gdf"
    base = data_dir

    assert (
        not overflowdetection
    ), "load_gdf2mat_feat_mne does not support overflowdetection..."

    # Load mat files
    data_path = (
        base
        + "/gdf2mat_overflowdetection_off/"
        + filename
        + "_overflowdetection_off.mat"
    )
    label_path = base + "/true_labels/" + filename[:4] + ".mat"

    session_data = loadmat(data_path, squeeze_me=False)
    label_data = loadmat(label_path, squeeze_me=False)

    gdf_data_path = base + "/" + filename[:4] + ".gdf"
    raw_gdf = mne.io.read_raw_gdf(gdf_data_path, stim_channel="auto")
    raw_gdf.load_data()

    # Parse data
    s = raw_gdf.get_data().T  # cnt -> tnc
    assert np.allclose(
        s * 1e6, session_data["s"]
    ), "mne and loadmat loaded different singal..."
    h = session_data["h"]  # header
    labels = label_data["classlabel"]  # true label

    h_names = h[0][0].dtype.names  # header is structured array
    origin_filename = h["FileName"][0, 0][0]
    train_labels = h["Classlabel"][0][0]  # For Evaluation data, it is filled with NaN.
    artifacts = h["ArtifactSelection"][0, 0]

    events = h["EVENT"][0, 0][0, 0]  # void
    typ = events["TYP"]
    pos = events["POS"]
    fs = events["SampleRate"].squeeze()
    dur = events["DUR"]

    # http://www.bbci.de/competition/iv/desc_2a.pdf
    typ2desc = {
        276: "Idling EEG (eyes open)",
        277: "Idling EEG (eyes closed)",
        768: "Start of a trial",
        769: "Cue onset left (class 1)",
        770: "Cue onset right (class 2)",
        771: "Cue onset foot (class 3)",
        772: "Cue onset tongue (class 4)",
        783: "Cue unknown",
        1024: "Eye movements",
        32766: "Start of a new run",
    }

    # 출처... 아마... brain decode...
    ch_names = [
        "Fz",
        "FC3",
        "FC1",
        "FCz",
        "FC2",
        "FC4",
        "C5",
        "C3",
        "C1",
        "Cz",
        "C2",
        "C4",
        "C6",
        "CP3",
        "CP1",
        "CPz",
        "CP2",
        "CP4",
        "P1",
        "Pz",
        "P2",
        "POz",
        "EOG-left",
        "EOG-central",
        "EOG-right",
    ]

    assert filename[:4] == origin_filename[:4]
    if verbose:
        print("- filename:", filename)
        print("- load data from:", data_path)
        print("\t- original fileanme:", origin_filename)
        print("- load label from:", label_path)
        print("- shape of s", s.shape)  # (time, 25 channels),
        print("- shape of labels", labels.shape)  # (288 trials)

    data = {
        "s": s,
        "h": h,
        "labels": labels,
        "filename": filename,
        "artifacts": artifacts,
        "typ": typ,
        "pos": pos,
        "fs": fs,
        "dur": dur,
        "typ2desc": typ2desc,
        "ch_names": ch_names,
    }
    return data


@verbose_func_name
def s_to_cnt(data, verbose=False):
    data = deepcopy(data)
    assert ("s" in data.keys()) and ("cnt" not in data.keys())
    data["cnt"] = data.pop("s").T

    if verbose:
        print("- shape of cnt:", data["cnt"].shape)
    return data


@verbose_func_name
def rerange_label_from_0(data, verbose=False):
    data = deepcopy(data)
    data["labels"] = data["labels"] - 1
    assert np.array_equal(np.unique(data["labels"]), [0, 1, 2, 3])

    if verbose:
        print("- unique labels:", np.unique(data["labels"]))
    return data


@verbose_func_name
def rerange_pos_from_0(data, verbose=False):
    """
    In matlab, index starts from 1.
    In python, index starts from 0.
    To adapt index type data, subtract 1 from it.
    """
    data = deepcopy(data)
    data["pos"] = data["pos"] - 1
    assert data["pos"].min() == 0

    if verbose:
        print("- initial value:", data["pos"][0])
        print("- minimum value:", np.min(data["pos"]))
    return data


# 2. Preprocessing
#     1. drop EOG channels
#     2. replace break with mean
#     3. scaling (microvolt)
#     4. bandpass 4-38Hz (butterworth 3rd order)
#     5. exponential running standardization (init_block_size=1000, factor_new=1e-3)
#     6. epoch (cue-0.5ms ~ cue+4ms)
#     * no rejection

# In[2]:


import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter
from braindecode.datautil import (
    exponential_moving_standardize,  # moving은 최신, running은 예전 꺼. axis가 달라서 중요함!
)


@verbose_func_name
def drop_eog_from_cnt(data, verbose=False):
    assert (data["cnt"].shape[0] == 25) and (
        len(data["ch_names"]) == 25
    ), "the number of channels is not 25..."
    data = deepcopy(data)
    data["cnt"] = data["cnt"][0:22]
    data["ch_names"] = data["ch_names"][0:22]

    if verbose:
        print("- shape of cnt:", data["cnt"].shape)
    return data


@verbose_func_name
def replace_break_with_mean(data, verbose=False):
    data = deepcopy(data)
    cnt = data["cnt"]
    for i_chan in range(cnt.shape[0]):
        this_chan = cnt[i_chan]
        cnt[i_chan] = np.where(this_chan == np.min(this_chan), np.nan, this_chan)
        mask = np.isnan(cnt[i_chan])
        chan_mean = np.nanmean(cnt[i_chan])
        cnt[i_chan, mask] = chan_mean
    data["cnt"] = cnt
    assert not np.any(np.isnan(cnt)), "nan remains in cnt.."

    if verbose:
        print("- min of cnt:", np.min(cnt))
    return data


@verbose_func_name
def change_scale(data, factor, channels="all", verbose=False):
    """
    Args
    ----
    data : dict
    factor : float
    channels : list of int, or int
    verbose : bool
    """
    data = deepcopy(data)
    if channels == "all":
        channels = list(range(data["cnt"].shape[0]))
    elif isinstance(channels, int):
        channels = [channels]

    assert hasattr(channels, "__len__"), "channels should be list or int..."

    assert (max(channels) <= data["cnt"].shape[0]) and (
        min(channels) >= 0
    ), "channel index should be between 0 and #channel of data..."

    assert ("s" not in data.keys()) and ("cnt" in data.keys())

    data["cnt"][channels, :] = data["cnt"][channels, :] * factor

    if verbose:
        print("- applied channels:", channels)
        print("- factor :", factor)
        print("- maximum value:", np.max(data["cnt"]))
        print("- minimum value:", np.min(data["cnt"]))
    return data


def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_lowpass(highcut, fs, order):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype="lowpass")
    return b, a


def butter_highpass(lowcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype="highpass")
    return b, a


@verbose_func_name
def butter_bandpass_filter(data, lowcut=0, highcut=0, order=3, axis=-1, verbose=False):
    assert (lowcut != 0) or (
        highcut != 0
    ), "one of lowcut and highcut should be not 0..."
    data = deepcopy(data)
    fs = data["fs"]

    if lowcut == 0:
        print("banpass changes into lowpass " "since lowcut is 0 ...")
        b, a = butter_lowpass(highcut, fs, order)
    elif highcut == 0:
        print("bandpass changes into highpass " "since highcut is 0 ...")
        b, a = butter_highpass(lowcut, fs, order)
    else:
        b, a = butter_bandpass(lowcut, highcut, fs, order)

    data["cnt"] = lfilter(b, a, data["cnt"], axis=axis)
    if verbose:
        if lowcut == 0:
            print(f"- lowpass : {highcut}Hz")
        elif highcut == 0:
            print(f"- highpass : {lowcut}Hz")
        else:
            print(f"- {lowcut}-{highcut}Hz")
        print(f"- order {order}")
        print(f"- fs {fs}Hz")
    return data


@verbose_func_name
def exponential_moving_standardize_from_braindecode(
    data, factor_new, init_block_size, eps=1e-4, verbose=False
):
    """
    for latest braindecode version...
    exponential_moving_standardize takes cnt (time, channel)
    """
    data = deepcopy(data)
    before_mean = np.mean(data["cnt"], axis=1)
    data["cnt"] = exponential_moving_standardize(
        data["cnt"], factor_new=factor_new, init_block_size=init_block_size, eps=eps
    )
    assert np.all(before_mean != np.mean(data["cnt"], axis=1))
    if verbose:
        print("- factor_new", factor_new)
        print("- init_block_size", init_block_size)
        print("- mean before standarization")
        print(before_mean)
        print("- mean after  standarization")
        print(np.mean(data["cnt"], axis=1))
    return data


# @verbose_func_name
# def exponential_running_standardize_from_braindecode(data,
#                          factor_new,
#                          init_block_size,
#                          eps=1e-4,
#                          verbose=False):
#     """
#     for outdated braindecode version...
#     exponential_running_standardize takes tnc (time, channel)
#     """
#     data = deepcopy(data)
#     before_mean = np.mean(data["cnt"], axis=1)
#     data["cnt"] = exponential_running_standardize(
#         data["cnt"].T,
#         factor_new=factor_new,
#         init_block_size=init_block_size,
#         eps=eps
#     ).T
#     assert np.all(before_mean != np.mean(data["cnt"], axis=1))
#     if verbose:
#         print("- factor_new", factor_new)
#         print("- init_block_size", init_block_size)
#         print("- mean before standarization")
#         print(before_mean)
#         print("- mean after  standarization")
#         print(np.mean(data["cnt"], axis=1))
#     return data


@verbose_func_name
def epoch_X_y_from_data(data, start_sec_offset, stop_sec_offset, verbose=False):
    """
    Args
    ----
    data : dict
        It can be obtained by load_gdf2mat and s_to_cnt functions.
    start_sec_offset : int
    stop_sec_offset : int
    verbose : bool

    Return
    ------
    X : 3d array (n_trials, n_channels, time)
    y : 2d array (n_trials, 1)

    NOTE
    ----
    The base of offset is 'start of a trial onset'.
    NOT based on 'cue onset'. if you want to use offset
    based on 'cue onset', add 2 sec to start_sec_offset
    and stop_sec_offset.
    """
    cnt = data["cnt"]
    pos = data["pos"]
    typ = data["typ"]
    fs = data["fs"]

    start_onset = pos[typ == 768]  # start of a trial
    trials = []
    for i, onset in enumerate(start_onset):
        trials.append(
            cnt[
                0:22,
                int(onset + start_sec_offset * fs) : int(onset + stop_sec_offset * fs),
            ]  # start of a trial + 1.5 ~ 6
        )
    X = np.array(trials)  # trials, channels, time
    y = data["labels"]

    if verbose:
        print("- From : start of a trial onset +", start_sec_offset, "sec")
        print("- To   : start of a trial onset +", stop_sec_offset, "sec")
        print("- shape of X", X.shape)
        print("- shape of y", y.shape)

    return X, y


def augment_by_cropping(X, y, crop_size, verbose=False):
    """ Augmentation by cropping given data

    Args:
        X (numpy.ndarray): shape of (n_trials, n_channels, n_time_samples)
        y (numpy.ndarray): shape of (n_trials, 1)
        crop_size (int): crop window size
    Returns:
        new_X (numpy.ndarray): shape of (n_trials, n_crops, n_channels, crop_size)
        new_y (numpy.ndarray): shape of (n_trials, n_crops, 1)
    """
    assert len(X) == len(y)
    _, _, n_time_samples = X.shape
    assert n_time_samples >= crop_size, "given time samples should be larger than crop_size"
    
    n_slices = n_time_samples - crop_size + 1
    new_X = []
    new_y = []
    for this_X, this_y in zip(X, y):
        this_new_X, this_new_y = [], []
        for i in range(n_slices):
            this_new_X.append(this_X[:, i:i+crop_size])
            this_new_y.append(this_y)
        this_new_X = np.stack(this_new_X, axis=0) # (n_crops, n_channels, crop_size)
        this_new_y = np.stack(this_new_y, axis=0) # (n_crops, 1)
        new_X.append(this_new_X)
        new_y.append(this_new_y)
    new_X = np.stack(new_X, axis=0) # (n_trials, n_crops, n_channels, crop_size)
    new_y = np.stack(new_y, axis=0) # (n_trials, n_crops, 1)
    
    assert len(new_X) == len(new_y)
    if verbose:
        print("- crop_size", crop_size)
        print("- shape of X", new_X.shape)
        print("- shape of y", new_y.shape)
    return new_X, new_y


def transform_augmented_2d_to_3d(X_aug_2d, verbose=False):
    """ Transform 2d augmented data into 3d representation.

    Args:
        X_aug_2d (numpy.ndarray): shape of (n_trials, n_crops, n_channels, n_time_samples)
    Return:
        X_aug_3d (numpy.ndarray): shape of (n_trials, n_crops, height, width, n_time_samples)
    """
    n_trials, n_crops, n_channels, n_time_samples = X_aug_2d.shape
    assert n_channels == 22, "currently only support 22 channels"
    X_aug_3d = np.zeros(shape=(n_trials, n_crops, 6, 7, n_time_samples))
    
    X_aug_3d[:, :, 0, 3, :] = X_aug_2d[:, :, 0, :]
    
    X_aug_3d[:, :, 1, 1, :] = X_aug_2d[:, :, 1, :]
    X_aug_3d[:, :, 1, 2, :] = X_aug_2d[:, :, 2, :]
    X_aug_3d[:, :, 1, 3, :] = X_aug_2d[:, :, 3, :]
    X_aug_3d[:, :, 1, 4, :] = X_aug_2d[:, :, 4, :]
    X_aug_3d[:, :, 1, 5, :] = X_aug_2d[:, :, 5, :]
    
    X_aug_3d[:, :, 2, 0, :] = X_aug_2d[:, :, 6, :]
    X_aug_3d[:, :, 2, 1, :] = X_aug_2d[:, :, 7, :]
    X_aug_3d[:, :, 2, 2, :] = X_aug_2d[:, :, 8, :]
    X_aug_3d[:, :, 2, 3, :] = X_aug_2d[:, :, 9, :]
    X_aug_3d[:, :, 2, 4, :] = X_aug_2d[:, :, 10, :]
    X_aug_3d[:, :, 2, 5, :] = X_aug_2d[:, :, 11, :]
    X_aug_3d[:, :, 2, 6, :] = X_aug_2d[:, :, 12, :]
    
    X_aug_3d[:, :, 3, 1, :] = X_aug_2d[:, :, 13, :]
    X_aug_3d[:, :, 3, 2, :] = X_aug_2d[:, :, 14, :]
    X_aug_3d[:, :, 3, 3, :] = X_aug_2d[:, :, 15, :]
    X_aug_3d[:, :, 3, 4, :] = X_aug_2d[:, :, 16, :]
    X_aug_3d[:, :, 3, 5, :] = X_aug_2d[:, :, 17, :]
    
    X_aug_3d[:, :, 4, 2, :] = X_aug_2d[:, :, 18, :]
    X_aug_3d[:, :, 4, 3, :] = X_aug_2d[:, :, 19, :]
    X_aug_3d[:, :, 4, 4, :] = X_aug_2d[:, :, 20, :]
    
    X_aug_3d[:, :, 5, 3, :] = X_aug_2d[:, :, 21, :]
    
    if verbose:
        print("- shape of X_aug_3d", X_aug_3d.shape)
    return X_aug_3d


def subtract_augmented_3d_mean(X):
    """ Subtract mean of each 3d tensor from the corresponding 3d tensor

    Args:
        X (numpy.ndarray): shape of (n_trials, n_crops, height, width, n_time_samples)
    """
    assert X.ndim == 5
    return X - X.mean(axis=(2, 3, 4))[:,:,np.newaxis,np.newaxis,np.newaxis]


# 3. Split
#     - split train into train and validation (8:2)

# In[3]:


@verbose_func_name
def split_train_val(X, y, val_ratio, verbose=False):
    assert (val_ratio < 1) and (val_ratio > 0), "val_raion not in (0, 1)"
    val_size = round(len(y) * val_ratio)
    X_tr, y_tr = X[:-val_size], y[:-val_size]
    X_val, y_val = X[-val_size:], y[-val_size:]
    assert (len(X_tr) == len(y_tr)) and (
        len(X_val) == len(y_val)
    ), "each pair of X and y should have same number of trials..."
    assert len(X) == len(X_tr) + len(
        X_val
    ), "sum of number of splited trials should equal number of unsplited trials"
    if verbose:
        print("- shape of X_tr", X_tr.shape)
        print("- shape of y_tr", y_tr.shape)
        print("- shape of X_val", X_val.shape)
        print("- shape of y_val", y_val.shape)
    return X_tr, y_tr, X_val, y_val


# 4. Crop (bunch of crops)
#     - input_time_length: 1000 samples
#     * augmentation effect (twice)

# In[4]:


import torch


# class TrialDataset:
#     def __init__(self, X, y, verbose=False):
#         assert len(X) == len(y), "X and y should have same length..."
#         self.X = X
#         self.y = y
#         self.trial_inds = torch.arange(len(y))

#         if verbose:
#             print("\nTrials")
#             print("- shape of X", self.X.shape)
#             print("- shape of y", self.y.shape)
#             print("- shape of trial_inds", self.trial_inds.shape)

#     def __getitem__(self, cur_ind):
#         return self.X[cur_ind], self.y[cur_ind], self.trial_inds[cur_ind]

#     def __len__(self):
#         return len(self.y)


class TrainingCropped3dDataset:
    def __init__(self, X, y, crop_stride, verbose=False):
        assert len(X) == len(y), "X and y should have same length..."
        assert X.ndim == 6, "X should be a tensor of shape (n_trials, n_crops, n_dim, height, width, crop_size"
        
        X = X[:, ::crop_stride, :, :, :, :]
        n_trials, m_crops, n_dim, height, width, crop_size = X.shape
        
        self.X = X.view(n_trials * m_crops, n_dim, height, width, crop_size)
        self.y = y.view(n_trials * m_crops, 1)
        self.trial_inds = (
            torch.arange(n_trials).view(-1,1) * torch.ones(size=(n_trials, m_crops))
        ).view_as(self.y).to(self.y.device)

        if verbose:
            print("\nTrials")
            print("- shape of X", self.X.shape)
            print("- shape of y", self.y.shape)
            print("- shape of trial_inds", self.trial_inds.shape)

    def __getitem__(self, cur_ind):
        return self.X[cur_ind], self.y[cur_ind], self.trial_inds[cur_ind]

    def __len__(self):
        return len(self.y)


class EvaluationCropped3dDataset:
    def __init__(self, X, y, crop_stride, verbose=False):
        assert len(X) == len(y), "X and y should have same length..."
        assert X.ndim == 6, "X should be a tensor of shape (n_trials, n_crops, n_dim, height, width, crop_size)"
        n_trials, _, _, _, _, _ = X.shape
        
        self.X = X[:, ::crop_stride, :, :, :, :]
        self.y = y[:, ::crop_stride, :]
        self.trial_inds = torch.arange(n_trials)

        if verbose:
            print("\nTrials")
            print("- shape of X", self.X.shape)
            print("- shape of y", self.y.shape)
            print("- shape of trial_inds", self.trial_inds.shape)

    def __getitem__(self, cur_ind):
        return self.X[cur_ind], self.y[cur_ind], self.trial_inds[cur_ind]

    def __len__(self):
        return len(self.y)


# 5. Data loader

# In[5]:


from torch.utils.data import DataLoader


# 6. Model
#     - ShallowNet
#     - to_dense_prediction_model

# In[6]:

import torch
from torch import nn
from torch.nn import init
from torch.functional import F
from collections import OrderedDict
from torch.nn.init import xavier_uniform_


class Square(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x ** 2

class Abs(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.abs(x)

class Log(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.log(x)


class Squeeze(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.squeeze(x, dim=3)
        return torch.squeeze(x, dim=2)


class TransposeTimeChannel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 1, 3, 2).contiguous()

## In[7]:


from torch import nn
from torch.nn import init
from collections import OrderedDict
from braindecode.models.util import to_dense_prediction_model

        
class Branch3DCNN(nn.Module):
    def __init__(self, mode, use_xavier_initialization=True, input_shape=None, verbose=False):
        super().__init__()
        if use_xavier_initialization is True:
            assert input_shape is not None
            
        assert mode in ["srf","mrf","lrf"]
        if mode == "srf":
            kernel_size = 1
            stride = 1
            padding_1 = (1,1,0)
            padding_2 = (0,0,0)
            flatten_elements = 60 * 64
        elif mode == "mrf":
            kernel_size = 3
            stride = 2
            padding_1 = (1,1,1)
            padding_2 = (0,0,1)
            flatten_elements = 15 * 64
        elif mode == "lrf":
            kernel_size = 5
            stride = 4
            padding_1 = (1,1,1)
            padding_2 = (0,0,1)
            flatten_elements = 4 * 64
        
        self.model = nn.Sequential(OrderedDict([
            ("conv_1", nn.Conv3d(16,  32, kernel_size=(2, 2, kernel_size), stride=(2, 2, stride), padding=padding_1)),
            ("bn_1", nn.BatchNorm3d(32)),
            ("elu_1", nn.ELU()),
            ("conv_2", nn.Conv3d(32, 64, kernel_size=(2, 2, kernel_size), stride=(2, 2, stride), padding=padding_2)),
            ("bn_2", nn.BatchNorm3d(64)),
            ("elu_2", nn.ELU()),
            ("flatten", nn.Flatten()),
            ("dense_1", nn.Linear(flatten_elements, 32)),
            ("bn_1d_1", nn.BatchNorm1d(32)),
            ("relu_1", nn.ReLU()),
            ("dense_2", nn.Linear(32, 32)),
            ("bn_1d_2", nn.BatchNorm1d(32)),
            ("relu_2", nn.ReLU()),
            ("dense_3", nn.Linear(32, 4)),
            ("softmax", nn.Softmax(dim=-1))
        ]))
        
        if use_xavier_initialization is True:
            # assert len(input_shape) == 5, "input_shape should be tuple of (batch_size, n_dim, height, width, time_samples)"
            # dummy = self.model.conv_1.weight.new_zeros(input_shape) 
            # with torch.no_grad():
            #     self.model(dummy) # to make weight of LazyLinear layer...
            self._xavier_initialization()
            
        if verbose is True:
            print(self)
        
    def forward(self, x):
        return self.model(x)
    
    def _xavier_initialization(self):
        init.xavier_uniform_(self.model.conv_1.weight, gain=1.)
        init.constant_(self.model.conv_1.bias, 0.)
        
        init.xavier_uniform_(self.model.conv_2.weight, gain=1.)
        init.constant_(self.model.conv_2.bias, 0.)
        
        init.xavier_uniform_(self.model.dense_1.weight, gain=1.)
        init.constant_(self.model.dense_1.bias, 0.)
        
        init.xavier_uniform_(self.model.dense_2.weight, gain=1.)
        init.constant_(self.model.dense_2.bias, 0.)
        
        init.xavier_uniform_(self.model.dense_3.weight, gain=1.)
        init.constant_(self.model.dense_3.bias, 0.)


class MultiBranch3DCNN(nn.Module):
    def __init__(self, use_xavier_initialization=True, input_shape=None, verbose=False):
        super().__init__()
        self.shared_layer = nn.Sequential(OrderedDict([
            ("conv", nn.Conv3d(1, 16, kernel_size=(3,3,5), stride=(2,2,4), padding=(1,0,1))),
            ("bn", nn.BatchNorm3d(16)),
            ("elu", nn.ELU()),
        ]))
        self.srf = Branch3DCNN(mode="srf", use_xavier_initialization=use_xavier_initialization, input_shape=input_shape)
        self.mrf = Branch3DCNN(mode="mrf", use_xavier_initialization=use_xavier_initialization, input_shape=input_shape)
        self.lrf = Branch3DCNN(mode="lrf", use_xavier_initialization=use_xavier_initialization, input_shape=input_shape)
        
        if verbose is True:
            print(self)
        
    def forward(self, x):
        # x (batch_size, 1, 6, 7, 240) <- padding(1,0,1)
        x = self.shared_layer(x) # (batch_size, 16, 3, 3, 60)
        x_srf = self.srf(x)
        x_mrf = self.mrf(x)
        x_lrf = self.lrf(x)
        return x_srf + x_mrf + x_lrf
        

# 7. Learning strategy
#     - loss function : log softmax + NLLloss for each crop
#                       ** tied sample loss ? **
#     - optimizer : Adam
#     - evaluater for cropped learning
#     - early stop : 1. using training, no decrease on val acc (80 epoch) or max epoch 800
#                    2. using training and val, same training loss with val loss from first stop or max epoch 800.
#     - maxnorm

# In[8]:


import torch
from torch.functional import F


# def crop_ce_loss(outputs, labels):
#     """
#     Arg
#     ---
#     outputs : 3d Tensor (n_batch, n_classes, n_preds_per_input)
#     labels  : 2d Tensor (n_batch, 1)
#     """
#     assert outputs.dim() == 3
#     assert len(outputs) == len(labels)
#     n_batch, n_classes, n_preds_per_input = outputs.shape

#     out = outputs.permute(1, 0, 2)
#     out = out.reshape(n_classes, n_batch * n_preds_per_input)
#     out = out.permute(1, 0)

#     lab = labels * labels.new_ones(n_batch, n_preds_per_input)
#     lab = lab.reshape(n_batch * n_preds_per_input)
#     return F.cross_entropy(out, lab)


# def tied_sample_loss(outputs):
#     """
#     Arg
#     ---
#     outputs : 3d Tensor (n_batch, n_classes, n_preds_per_input)
#     """
#     assert outputs.dim() == 3
#     this_prob = F.softmax(outputs[:, :, :-1], dim=1)
#     next_prob = F.softmax(outputs[:, :, 1:], dim=1)

#     loss = torch.sum(
#         -torch.log(this_prob) * next_prob, dim=1
#     )  # sum over dim of n_classes
#     return torch.mean(loss)


# def trial_pred_from_crop_outputs(outputs):
#     """
#     Args
#     ----
#     outputs : 3d tensor (n_batch, n_classes, n_preds_per_input)

#     Return
#     ------
#     preds : 2d tensor (n_trials, 1)
#     """
#     assert outputs.dim() == 3
#     #     probs = torch.softmax(outputs, dim=1).mean(dim=2) # as I understood...
#     probs = F.log_softmax(outputs, dim=1).mean(dim=2)  # according to github code
#     return torch.argmax(probs, dim=1, keepdim=True)


def evaluate_using_dataloaders(
    model, dataloaders, which_learning, mode, prefix
):
    if which_learning == "trialwise":
        raise
    elif which_learning == "cropped":
        return evaluate_using_cropped_dataloaders(
            model, dataloaders, mode, prefix=prefix, 
        )


def evaluate_using_cropped_dataloaders(model, dataloaders, mode, prefix):
    if mode == "tr":
        return evaluate_using_cropped_training_dataloader(model, dataloaders["tr"], prefix)
    elif mode == "te":
        return evaluate_using_cropped_test_dataloader(model, dataloaders["te"], prefix)
        
    
def evaluate_using_cropped_training_dataloader(model, dataloader, prefix):
    model.eval()
    total_outputs = []
    total_labels = []
    with torch.no_grad():
        for inputs, labels, _ in dataloader:
            total_outputs.append(model(inputs))
            total_labels.append(labels)

        total_outputs = torch.cat(total_outputs, dim=0)
        total_labels = torch.cat(total_labels, dim=0)
        total_probs = torch.softmax(total_outputs, dim=1)
        total_preds = torch.argmax(total_probs, dim=1, keepdim=True)

        assert total_preds.shape == total_labels.shape
        total_corrects = (total_preds == total_labels)
        acc = torch.mean(total_corrects.float())
        ce_loss = F.cross_entropy(total_outputs, total_labels.flatten())

        return {
            f"{prefix}tr_acc": acc.item(),
            f"{prefix}tr_ce_loss": ce_loss.item(),
        }


def evaluate_using_cropped_test_dataloader(model, dataloader, prefix):
    model.eval()
    total_outputs = []
    total_labels = []
    n_trials = 0
    with torch.no_grad():
        for inputs, labels, _ in dataloader:
            batch_size, n_crops, n_dim, hegiht, width, crop_size = inputs.shape
            inputs = inputs.view(batch_size * n_crops, n_dim, hegiht, width, crop_size)
            labels = labels.view(batch_size * n_crops, 1)
            
            total_outputs.append(model(inputs))
            total_labels.append(labels)
            n_trials += batch_size
            
        trial_crop_outputs = torch.cat(total_outputs, dim=0).view(n_trials, n_crops, 4)
        trial_crop_labels = torch.cat(total_labels).view(n_trials, n_crops, 1)
        
        trial_crop_probs = torch.softmax(trial_crop_outputs, dim=2)
        trial_preds = torch.argmax(torch.sum(trial_crop_probs, dim=1), dim=1)
        
        trial_labels = trial_crop_labels[:,0,0]
        assert torch.equal(
            trial_labels.unsqueeze(1) * trial_labels.new_ones(size=(n_trials, n_crops)),
            trial_crop_labels.squeeze()
        )
        
        assert trial_preds.shape == trial_labels.shape
        trial_corrects = (trial_preds == trial_labels)
        acc = torch.mean(trial_corrects.float())
        
        total_outputs = torch.cat(total_outputs, dim=0)
        total_labels = torch.cat(total_labels, dim=0)
        ce_loss = F.cross_entropy(total_outputs, total_labels.flatten())

        return {
            f"{prefix}te_acc": acc.item(),
            f"{prefix}te_ce_loss": ce_loss.item(),
        }


class EarlyStopNoDecrease:
    def __init__(self, column_name, patient_epochs, min_decrease=1e-6):
        self.column_name = column_name
        self.patient_epochs = patient_epochs
        self.min_decrease = min_decrease
        self.best_epoch = 0
        self.best_value = 9e999

    def __call__(self, epoch_df):
        """
        Args
        ----
        epoch_df : pandas.DataFrame

        Return
        ------
        stop : bool
        """
        assert self.column_name in epoch_df.columns, f"{self.column_name} not in epoch_df..."
        this_epoch = epoch_df.index[-1]
        this_value = epoch_df[self.column_name].iloc[-1]
        if this_value < ((1 - self.min_decrease) * self.best_value):
            self.best_epoch = this_epoch
            self.best_value = this_value
        return (this_epoch - self.best_epoch) >= self.patient_epochs


class SaveBestModel:
    def __init__(self):
        self.best_val_acc = 0
        self.best_tr_ce_loss = 9e999
        self.saved_weight = None
        self.saved_optimizer = None
        self.saved_epoch = -1

    def if_best_val_acc(self, model, optimizer, epoch_df):
        this_val_acc = epoch_df["val_acc"].iloc[-1]
        if self.best_val_acc <= this_val_acc:
            self.best_val_acc = this_val_acc
            self.saved_weight = model.state_dict().copy()
            self.saved_optimizer = optimizer.state_dict().copy()
            self.saved_epoch = epoch_df.index[-1]
            print("  new best val acc:", this_val_acc)
            
    def if_best_tr_ce_loss(self, model, optimizer, epoch_df):
        this_tr_ce_loss = epoch_df["tr_ce_loss"].iloc[-1]
        if self.best_tr_ce_loss >= this_tr_ce_loss:
            self.best_tr_ce_loss = this_tr_ce_loss
            self.saved_weight = model.state_dict().copy()
            self.saved_optimizer = optimizer.state_dict().copy()
            self.saved_epoch = epoch_df.index[-1]
            print("  new best tr ce loss:", this_tr_ce_loss)

    def restore_best_model(self, model, optimizer, epoch_df):
        print("  model load weight saved at", self.saved_epoch, "epoch")
        model.load_state_dict(self.saved_weight.copy())
        optimizer.load_state_dict(self.saved_optimizer.copy())
        epoch_df.drop(range(self.saved_epoch + 1, len(epoch_df)), inplace=True)


# In[9]:


# def maxnorm(model):
#     last_weight = None
#     for name, module in list(model.named_children()):
#         if hasattr(module, "weight") and (
#             not module.__class__.__name__.startswith("BatchNorm")
#         ):
#             module.weight.data = torch.renorm(module.weight.data, 2, 0, maxnorm=2)
#             last_weight = module.weight
#     if last_weight is not None:
#         last_weight.data = torch.renorm(last_weight.data, 2, 0, maxnorm=0.5)

# ---

# ---

# In[ ]:


import argparse
import os
import logging
import sys

def exp(args):
    assert os.path.exists(args.result_dir)

    log = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s %(levelname)s : %(message)s",
        level=logging.DEBUG,
        stream=sys.stdout,
    )

    def print(*texts):
        texts = " ".join([str(_) for _ in texts])
        for line in texts.split("\n"):
            log.info(line)

    print("args:", args)

    # In[ ]:

    subject = args.subject
    data_dir = "/home/jinhyo/multi_class_motor_imagery/data/BCICIV_2a/gdf"

    order = 3
    start_sec_offset = args.start_sec_offset
    stop_sec_offset = args.stop_sec_offset

    save_name = args.save_name
    which_learning = args.which_learning
    device = args.device

    max_epochs = 800
    patient_epochs = 80

    assert which_learning in ["trialwise", "cropped"]

    # 1. Data load  & 2. Preprocessing

    # In[12]:

    print("\nLoad Train Data")
    data_tr = load_gdf2mat_feat_mne(
        subject=subject,
        train=True,
        data_dir=data_dir,
        overflowdetection=False,
        verbose=True,
    )
    data_tr = s_to_cnt(data_tr, verbose=True)
    data_tr = rerange_label_from_0(data_tr, verbose=True)
    data_tr = rerange_pos_from_0(data_tr, verbose=True)

    print("\nPreprocessing")
    data_tr = drop_eog_from_cnt(data_tr, verbose=True)
    data_tr = change_scale(data_tr, factor=1e06, channels="all", verbose=True)
    X_tr, y_tr = epoch_X_y_from_data(
        data_tr,
        start_sec_offset=start_sec_offset,
        stop_sec_offset=stop_sec_offset,
        verbose=True,
    )
    X_tr, y_tr = augment_by_cropping(
        X_tr, y_tr, args.crop_size, args.crop_stride
    ) # (n_trials, n_crops, n_channels, crop_size)
    X_tr = transform_augmented_2d_to_3d(X_tr) # (n_trials, n_crops, height, width, crop_size)
    X_tr = subtract_augmented_3d_mean(X_tr) # (n_trials, n_crops, height, width, crop_size)
    # In[13]:

    print("\nLoad Test Data")
    data_te = load_gdf2mat_feat_mne(
        subject=subject,
        train=False,
        data_dir=data_dir,
        overflowdetection=False,
        verbose=True,
    )
    data_te = s_to_cnt(data_te, verbose=True)
    data_te = rerange_label_from_0(data_te, verbose=True)
    data_te = rerange_pos_from_0(data_te, verbose=True)

    print("\nPreprocessing")
    data_te = drop_eog_from_cnt(data_te, verbose=True)
    data_te = change_scale(data_te, factor=1e06, channels="all", verbose=True)
    X_te, y_te = epoch_X_y_from_data(
        data_te,
        start_sec_offset=start_sec_offset,
        stop_sec_offset=stop_sec_offset,
        verbose=True,
    ) # X_te : (n_trials, n_channels, n_time_sample) 
    X_te, y_te = augment_by_cropping(
        X_te, y_te, args.crop_size, args.crop_stride
    ) # (n_trials, n_crops, n_channels, crop_size)
    X_te = transform_augmented_2d_to_3d(X_te) # (n_trials, n_crops, height, width, crop_size)
    X_te = subtract_augmented_3d_mean(X_te) #(n_trials, n_crops, height, width, crop_size)


    # 3. Cross validation (split)

    # In[14]:
    
    
    print("\nTo tensor")
    # channel first
    X_tr = torch.Tensor(X_tr[:, :, None, :, :, :]).to(device) # (n_trials, n_crops, 1, height, width, crop_size)
    y_tr= torch.Tensor(y_tr).long().to(device)
    print("- shape of X_tr:", X_tr.shape)
    print("- shape of y_tr:", y_tr.shape)
    X_te = torch.Tensor(X_te[:, :, None, :, :, :]).to(device) # (n_trials, n_crops, 1, height, width, crop_size)
    y_te = torch.Tensor(y_te).long().to(device)
    print("- shape of X_te:", X_te.shape)
    print("- shape of y_te:", y_te.shape)

    # In[15]:
    # from sklearn.model_selection import StratifiedKFold
    
    
    # skf = StratifiedKFold(n_splits=args.n_cv, shuffle=True, random_state=args.random_state)
    # y_trial = y_total[:,0,:]
    # for i_cv, (train_index, test_index) in enumerate(skf.split(X_total.cpu(), y_trial.cpu()), start=1):
    #     X_cv_tr = X_total[train_index, :, :, :, :, :] # (n_trainings, n_crops, n_dim, height, width, crop_size)
    #     y_cv_tr = y_total[train_index, :, :] # (n_trainings, n_crops, 1)
        
    #     X_cv_te = X_total[test_index, :, :, :, :, :] # (n_tests, n_crops, n_dim, height, width, crop_size)
    #     y_cv_te = y_total[test_index, :, :] # (n_tests, n_crops, n_dim, height, width, crop_size)
    
    # 4. Dataset

    # In[16]:
    
    print("\nDataset")
    if which_learning == "trialwise":
        raise
    elif which_learning == "cropped":
        print(f"\ntraining:")
        training_dataset_tr = TrainingCropped3dDataset(
            X=X_tr, 
            y=y_tr, 
            crop_stride=args.crop_stride,
            verbose=True,
        )
        
        # evaluation_dataset_tr = EvaluationCropped3dDataset(
        #     X=X_cv_tr, 
        #     y=y_cv_tr, 
        #     crop_stride=args.crop_stride_eval,
        #     verbose=True,
        # )

        print(f"\ntest: ")
        evaluation_dataset_te = EvaluationCropped3dDataset(
            X_te, 
            y_te,
            crop_stride=args.crop_stride_eval, 
            verbose=True,
        )

    # 5. Data loader

    # In[17]:
    dataloaders = {
        "tr": DataLoader(training_dataset_tr, batch_size=args.batch_size, shuffle=True),
        "te": DataLoader(evaluation_dataset_te, batch_size=args.batch_size, shuffle=False),
    }

    results = []
    for i_try in range(args.i_try_start, args.i_try_start + args.repeat):
        print("\n# TRY", i_try, "\n")
        # 6. Model

        # In[18]:
        model = MultiBranch3DCNN(use_xavier_initialization=args.use_xavier_initialization, input_shape=args.input_shape, verbose=True)

        model = model.to(device)

        # In[19]:

        # 7. Learning strategy

        # In[20]:

        class LossCase:
            def __init__(self, which_learning):
                self.which_learning = which_learning

            def __call__(self, outputs, labels):
                if self.which_learning == "trialwise":
                    raise

                elif self.which_learning == "cropped":
                    return F.cross_entropy(outputs, labels.flatten())
        # In[21]:

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_init)
        save_best_model = SaveBestModel()
        loss_function = LossCase(which_learning=which_learning)

        if args.use_early_stop is True:                
            early_stop_no_decrease = EarlyStopNoDecrease(
                column_name=args.early_stop_column, patient_epochs=args.patient_epochs
            )
        
        if args.use_lr_scheduler is True:
            lr_scheduler =  torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_gamma)
        
        print("\nTraining")
        epoch_df = pd.DataFrame()            
        for epoch in range(0, args.epoch):
            # Train
            model.train()
            for inputs, labels, _ in dataloaders["tr"]:  
                outputs = model(inputs)
                loss = loss_function(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Evaluation
            evaluation_tr = evaluate_using_dataloaders(
                model,
                dataloaders=dataloaders,
                which_learning=which_learning,
                mode="tr",
                prefix="",
            )
            evaluation_te = evaluate_using_dataloaders(
                model,
                dataloaders=dataloaders,
                which_learning=which_learning,
                mode="te",
                prefix="",
            )

            assert len(epoch_df) == epoch
            epoch_df = epoch_df.append(
                dict(
                    **evaluation_tr, 
                    **evaluation_te,
                ),
                ignore_index=True,
            )
            print("epoch", epoch)
            print(epoch_df.iloc[-1])
            print()
            
            save_best_model.if_best_tr_ce_loss(model, optimizer, epoch_df)
                
            if args.use_early_stop is True:
                if early_stop_no_decrease(epoch_df):
                    print("early stop !")
                    save_best_model.restore_best_model(model, optimizer, epoch_df)
                    break
                
            if args.use_lr_scheduler is True:
                if len(epoch_df) > 1:
                    if (epoch_df[args.lr_scheduler_column].iloc[-2] <= epoch_df[args.lr_scheduler_column].iloc[-1]):
                        lr_scheduler.step()
                        print("\nchange learning rate:", lr_scheduler.get_last_lr(), "\n")
                
        # In[23]:

        print("\nLast Epoch")
        print(epoch_df.iloc[-1])

        # In[24]:

        # epoch_df[["tr_acc", "val_acc", "te_acc"]].plot()

        # In[25]:

        # epoch_df[["tr_loss", "val_loss", "te_loss"]].plot()

        # In[26]:

        # epoch_df[["tr_ce_loss", "val_ce_loss", "te_ce_loss"]].plot()

        # In[27]:

        result_name = f"{args.result_dir}/{save_name}_subject{subject}_try{i_try}"
        epoch_df.to_csv(result_name + ".csv")
        torch.save(model.state_dict(), result_name + ".h5")

        # In[ ]:
        results.append(round(epoch_df["te_acc"].iloc[-1], 2))

    print(f"\n{args.n_cv} :{args.repeat} results (te_acc)")
    for i_try in range(1, 1 + args.repeat):
        print(f"{i_try} try : {results[i_try-1]}")
    print(f"mean : {np.mean(results):.2f} ± {np.std(results):.2f}")



# In[ ]:
if __name__ == "__main__":
    # def str2bool(v):
    #     if isinstance(v, bool):
    #         return v
    #     if v.lower() in ('yes', 'true', 't', 'y', '1'):
    #         return True
    #     elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    #         return False
    #     else: raise argparse.ArgumentTypeError('Boolean value expected.')

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--subject", required=True, type=int)
    # parser.add_argument("--lowcut", required=True, type=int,
    #                    help="0 or 4")
    # parser.add_argument("--which_model", required=True,
    #                    help="ShallowNet or DenseNet")
    # parser.add_argument("--which_learning", required=True,
    #                     help="trialwise or cropped")
    # parser.add_argument("--use_tied_loss", required=True, type=str2bool,
    #                    help="True or False")
    # parser.add_argument("--device", default="cuda",
    #                    help="cuda or cpu")
    # parser.add_argument("--result_dir", required=True)
    # parser.add_argument("--repeat", default=1, type=int)
    # args = parser.parse_args()
    class Args:
        subject = None
        # preprocessing
        start_sec_offset = 2.0
        stop_sec_offset = 3.25 + 0.002 # to make time sample as 313
        # augmentation
        crop_size = 240
        crop_stride = 1
        crop_stride_eval = 5
        # cross validation
        n_cv = 10
        random_state = 20211227
        # model
        use_xavier_initialization = True
        input_shape = (1, 16, 4, 4, 60)
        # training
        which_learning = "cropped"
        epoch = 500
        batch_size = 60
        device = "cuda:2"
        # early stop
        use_early_stop = True
        early_stop_column = "tr_ce_loss"
        patient_epochs = 20
        # learning rate scheduler
        use_lr_scheduler = True
        lr_init = 0.01
        lr_scheduler_column = "tr_ce_loss"
        lr_gamma = 0.1
        # save
        save_name = f"multi_branch_3d_cnn_split_competition"
        result_dir = __file__.split("/")[-1].split(".")[0]
        # experiment
        i_try_start = 1
        repeat = 10

    args = Args()
    for subject in range(1,10):
        args.subject = subject
        exp(args) 
