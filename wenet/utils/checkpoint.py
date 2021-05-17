# Copyright 2019 Mobvoi Inc. All Rights Reserved.
# Author: binbinzhang@mobvoi.com (Binbin Zhang)

import logging
import os
import re

import yaml
import torch


def load_checkpoint(model: torch.nn.Module, path: str) -> dict:
    if torch.cuda.is_available():
        logging.info('Checkpoint: loading from checkpoint %s for GPU' % path)
        checkpoint = torch.load(path)
    else:
        logging.info('Checkpoint: loading from checkpoint %s for CPU' % path)
        checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint)
    info_path = re.sub('.pt$', '.yaml', path)
    configs = {}
    if os.path.exists(info_path):
        with open(info_path, 'r') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
    return configs

def load_part_params(model: torch.nn.Module, path: str) -> dict:
    if torch.cuda.is_available():
        logging.info('Checkpoint: loading from checkpoint %s for GPU' % path)
        checkpoint = torch.load(path)
    else:
        logging.info('Checkpoint: loading from checkpoint %s for CPU' % path)
        checkpoint = torch.load(path, map_location='cpu')

    keys = ['encoder.encoders.' + str(i) for i in [0, 4, 7, 11]]
    # filter_keys = ['encoder.encoders.' + str(i) + '.' for i in range(12) if i not in [0, 4, 7, 11]]
    filter_keys = ['encoder.encoders.' + str(i) + '.' for i in range(12) if i not in [0,1,2,3]]
    print(filter_keys)

    params = {}
    for k, v in checkpoint.items():
        if k.startswith('decoder') or k.startswith(tuple(filter_keys)):
            continue
        params[k] = v
    print(len(params.keys()))
    params_new = {}
    for k, v in params.items():
        for i, x in enumerate(keys):
            if k.startswith(x):
                k = '.'.join(k.split('.')[0:2]) + '.' + str(i) + '.' + '.'.join(k.split('.')[3:])
                params_new[k] = v
                break
        params_new[k] = v
        # print(k, v.shape)
    print(len(params_new.keys()))

    model_dict = model.state_dict()
    for k, v in model_dict.items():
        if k in params_new.keys() and v.size() == params_new[k].size():
            model_dict[k] = params_new[k]
        if k in params_new.keys() and v.size() != params_new[k].size():
            assert len(v.shape) == len(params_new[k].shape)
            if len(v.shape) == 1:
                l1 = v.shape[0]
                l2 = params_new[k].shape[0]
                model_dict[k] = params_new[k][::l2 // l1]
            if len(v.shape) == 2:
                if v.shape[0] == params_new[k].shape[0]:
                    l1 = v.shape[1]
                    l2 = params_new[k].shape[1]
                    model_dict[k] = params_new[k][:, ::l2 // l1]
                if v.shape[1] == params_new[k].shape[1]:
                    l1 = v.shape[0]
                    l2 = params_new[k].shape[0]
                    model_dict[k] = params_new[k][::l2 // l1, :]
    model.load_state_dict(model_dict)
    return {}

# def load_part_params(model: torch.nn.Module, path: str) -> dict:
#     if torch.cuda.is_available():
#         logging.info('Checkpoint: loading from checkpoint %s for GPU' % path)
#         checkpoint = torch.load(path)
#     else:
#         logging.info('Checkpoint: loading from checkpoint %s for CPU' % path)
#         checkpoint = torch.load(path, map_location='cpu')
#
#     keys = ['encoder.encoders.' + str(i) for i in [0, 4, 7, 11]]
#     filter_keys = ['encoder.encoders.' + str(i) + '.' for i in range(12) if i not in [0, 4, 7, 11]]
#     print(filter_keys)
#
#     params = {}
#     for k, v in checkpoint.items():
#         if k.startswith('decoder') or k.startswith(tuple(filter_keys)) or 'feed_forward' in k:
#             continue
#         params[k] = v
#     print(len(params.keys()))
#     params_new = {}
#     for k, v in params.items():
#         for i, x in enumerate(keys):
#             if k.startswith(x):
#                 k = '.'.join(k.split('.')[0:2]) + '.' + str(i) + '.' + '.'.join(k.split('.')[3:])
#                 params_new[k] = v
#                 break
#         params_new[k] = v
#         # print(k, v.shape)
#     print(len(params_new.keys()))
#
#     model_dict = model.state_dict()
#     for k, v in model_dict.items():
#         if k in params_new.keys() and v.size() == params_new[k].size():
#             model_dict[k] = params_new[k]
#     model.load_state_dict(model_dict)
#     return {}

def save_checkpoint(model: torch.nn.Module, path: str, infos=None):
    '''
    Args:
        infos (dict or None): any info you want to save.
    '''
    logging.info('Checkpoint: save to checkpoint %s' % path)
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(state_dict, path)
    info_path = re.sub('.pt$', '.yaml', path)
    if infos is None:
        infos = {}
    with open(info_path, 'w') as fout:
        data = yaml.dump(infos)
        fout.write(data)
