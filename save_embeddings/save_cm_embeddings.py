from __future__ import absolute_import
import argparse
import json
import os
import pickle as pk
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
import importlib

# from aasist.data_utils import Dataset_ASVspoof2019_devNeval
from model import Model as LCNN
# from aasist.models.AASIST import Model as AASISTModel
# from ECAPATDNN.model import ECAPA_TDNN
from utils import load_parameters

import display as nii_warn
import default_data_io as nii_dset
import data_io_conf as nii_dconf
import list_tools as nii_list_tool
import config_parse as nii_config_parse
import arg_parse as nii_arg_parse
import op_manager as nii_op_wrapper
import nn_manager as nii_nn_wrapper
import startup_config as nii_startup

# list of dataset partitions
SET_PARTITION = ["trn", "dev", "eval"]

# list of countermeasure(CM) protocols
SET_CM_PROTOCOL = {
    "trn": "/content/PA/PA/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.train.trn.txt",
    "dev": "/content/PA/PA/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.dev.trl.txt",
    "eval": "/content/PA/PA/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.dev.trl.txt",
}

# directories of each dataset partition
SET_DIR = {
    "trn": "/content/PA/PA/ASVspoof2019_PA_train",
    "dev": "/content/PA/PA/ASVspoof2019_PA_dev",
    "eval": "/content/PA/PA/ASVspoof2019_PA_eval",
}

# enrolment data list for speaker model calculation
# each speaker model comprises multiple enrolment utterances
SET_TRN = {
    "dev": [
        "/content/PA/PA/ASVspoof2019_PA_asv_protocols/ASVspoof2019.PA.asv.dev.female.trn.txt",
        "/content/PA/PA/ASVspoof2019_PA_asv_protocols/ASVspoof2019.PA.asv.dev.male.trn.txt",
    ],
    "eval": [
        "/content/PA/PA/ASVspoof2019_PA_asv_protocols/ASVspoof2019.PA.asv.eval.female.trn.txt",
        "/content/PA/PA/ASVspoof2019_PA_asv_protocols/ASVspoof2019.PA.asv.eval.male.trn.txt",
    ],
}


# def save_embeddings(
#     set_name, cm_embd_ext, asv_embd_ext, device
# ):
def save_embeddings(
    set_name, cm_embd_ext, test_set, device
):
    meta_lines = open(SET_CM_PROTOCOL[set_name], "r").readlines()
    utt2spk = {}
    utt_list = []
    for line in meta_lines:
        tmp = line.strip().split(" ")

        spk = tmp[0]
        utt = tmp[1]

        if utt in utt2spk:
            print("Duplicated utt error", utt)

        utt2spk[utt] = spk
        utt_list.append(utt)

    base_dir = SET_DIR[set_name]
    # dataset = Dataset_ASVspoof2019_devNeval(utt_list, Path(base_dir))
    test_data_loader = test_set.get_loader()
    test_seq_num = test_set.get_seq_num()
    test_set.print_info()

    cm_emb_dic = {}
    # asv_emb_dic = {}

    print("Getting embeddings from set %s..." % (set_name))
    cm_embd_ext.extracting = True

    for batch_x, data_tar, data_info, idx_orig in tqdm(test_data_loader):
        batch_x = batch_x.to(device)
        with torch.no_grad():
            batch_cm_emb = cm_embd_ext(batch_x, data_info)
            batch_cm_emb = batch_cm_emb.detach().cpu().numpy()
            # batch_asv_emb = asv_embd_ext(batch_x, aug=False).detach().cpu().numpy()

        for k, cm_emb in zip(data_info, batch_cm_emb):
            cm_emb_dic[k] = cm_emb
        #     asv_emb_dic[k] = asv_emb

    os.makedirs("embeddings", exist_ok=True)
    with open( "embeddings/cm_embd_%s.pk" % (set_name), "wb") as f:
        pk.dump(cm_emb_dic, f)
    # with open("embeddings/asv_embd_%s.pk" % (set_name), "wb") as f:
    #     pk.dump(asv_emb_dic, f)


# def save_models(set_name, asv_embd_ext, device):
#     utt2spk = {}
#     utt_list = []

#     for trn in SET_TRN[set_name]:
#         meta_lines = open(trn, "r").readlines()

#         for line in meta_lines:
#             tmp = line.strip().split(" ")

#             spk = tmp[0]
#             utts = tmp[1].split(",")

#             for utt in utts:
#                 if utt in utt2spk:
#                     print("Duplicated utt error", utt)

#                 utt2spk[utt] = spk
#                 utt_list.append(utt)

#     base_dir = SET_DIR[set_name]
#     dataset = Dataset_ASVspoof2019_devNeval(utt_list, Path(base_dir))
#     loader = DataLoader(
#         dataset, batch_size=30, shuffle=False, drop_last=False, pin_memory=True
#     )
#     asv_emb_dic = {}

#     print("Getting embedgins from set %s..." % (set_name))

#     for batch_x, key in tqdm(loader):
#         batch_x = batch_x.to(device)
#         with torch.no_grad():
#             batch_asv_emb = asv_embd_ext(batch_x, aug=False).detach().cpu().numpy()

#         for k, asv_emb in zip(key, batch_asv_emb):
#             utt = k
#             spk = utt2spk[utt]
            

#             if spk not in asv_emb_dic:
#                 asv_emb_dic[spk] = []

#             asv_emb_dic[spk].append(asv_emb)

#     for spk in asv_emb_dic:
#         asv_emb_dic[spk] = np.mean(asv_emb_dic[spk], axis=0)

#     with open("embeddings/spk_model.pk_%s" % (set_name), "wb") as f:
#         pk.dump(asv_emb_dic, f)



def main():
    args = nii_arg_parse.f_args_parsed()

    # 
    nii_warn.f_print_w_date("Start program", level='h')
    nii_warn.f_print("Load module: %s" % (args.module_config))
    nii_warn.f_print("Load module: %s" % (args.module_model))
    prj_conf = importlib.import_module(args.module_config)
    prj_model = importlib.import_module(args.module_model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))

    # with open(args.aasist_config, "r") as f_json:
    #     config = json.loads(f_json.read())

    prj_conf = importlib.import_module('config')

    params = {'batch_size':  args.batch_size,
                  'shuffle': False,
                  'num_workers': args.num_workers}
        
    if type(prj_conf.test_list) is list:
        t_lst = prj_conf.test_list
    else:
        t_lst = nii_list_tool.read_list_from_text(prj_conf.test_list)
    test_set = nii_dset.NIIDataSetLoader(
        prj_conf.test_set_name, \
        t_lst, \
        prj_conf.test_input_dirs,
        prj_conf.input_exts, 
        prj_conf.input_dims, 
        prj_conf.input_reso, 
        prj_conf.input_norm,
        prj_conf.test_output_dirs, 
        prj_conf.output_exts, 
        prj_conf.output_dims, 
        prj_conf.output_reso, 
        prj_conf.output_norm,
        './',
        params = params,
        truncate_seq= None,
        min_seq_len = None,
        save_mean_std = False,
        wav_samp_rate = prj_conf.wav_samp_rate,
        global_arg = args)
        
    # initialize model
    cm_embd_ext = LCNN(test_set.get_in_dim(), \
                            test_set.get_out_dim(), \
                            args, prj_conf)

    # cm_embd_ext = LCNN(in_dim=1, out_dim=1, args=args, prj_conf=prj_conf, mean_std=None).to(device)
    load_parameters(cm_embd_ext.state_dict(), args.trained_model)
    cm_embd_ext.to(device)
    cm_embd_ext.eval()
    print("Loaded dataset and model")

    # asv_embd_ext = ECAPA_TDNN(C=1024)
    # load_parameters(asv_embd_ext.state_dict(), args.ecapa_weight)
    # asv_embd_ext.to(device)
    # asv_embd_ext.eval()

    for set_name in SET_PARTITION:
        save_embeddings(
            set_name,
            cm_embd_ext,
            test_set,
            device)
            # asv_embd_ext,
        # if set_name == "trn":
        #     continue
        # save_models(set_name, asv_embd_ext, device)


if __name__ == "__main__":
    main()
