import argparse
import json
import os
import pickle as pk
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from aasist.data_utils import Dataset_ASVspoof2019_devNeval
from ECAPATDNN.model import ECAPA_TDNN
from utils import load_parameters

# list of dataset partitions
SET_PARTITION = ["trn", "dev", "eval"]

# list of countermeasure(CM) protocols
SET_CM_PROTOCOL = {
    "trn": "./PA/PA/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.train.trn.txt",
    "dev": "./PA/PA/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.dev.trl.txt",
    "eval": "./PA/PA/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.dev.trl.txt",
}

# directories of each dataset partition
SET_DIR = {
    "trn": "./PA/PA/ASVspoof2019_PA_train",
    "dev": "./PA/PA/ASVspoof2019_PA_dev",
    "eval": "./PA/PA/ASVspoof2019_PA_eval",
}

# enrolment data list for speaker model calculation
# each speaker model comprises multiple enrolment utterances
SET_TRN = {
    "dev": [
        "./PA/PA/ASVspoof2019_PA_asv_protocols/ASVspoof2019.PA.asv.dev.female.trn.txt",
        "./PA/PA/ASVspoof2019_PA_asv_protocols/ASVspoof2019.PA.asv.dev.male.trn.txt",
    ],
    "eval": [
        "./PA/PA/ASVspoof2019_PA_asv_protocols/ASVspoof2019.PA.asv.eval.female.trn.txt",
        "./PA/PA/ASVspoof2019_PA_asv_protocols/ASVspoof2019.PA.asv.eval.male.trn.txt",
    ],
}


def save_embeddings(
    set_name, asv_embd_ext, device
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
    dataset = Dataset_ASVspoof2019_devNeval(utt_list, Path(base_dir))
    loader = DataLoader(
        dataset, batch_size=30, shuffle=False, drop_last=False, pin_memory=True
    )

    asv_emb_dic = {}

    print("Getting embeddings from set %s..." % (set_name))

    for batch_x, key in tqdm(loader):
        batch_x = batch_x.to(device)
        with torch.no_grad():
            batch_asv_emb = asv_embd_ext(batch_x, aug=False).detach().cpu().numpy()

        for k, asv_emb in zip(key, batch_asv_emb):
            asv_emb_dic[k] = asv_emb

    os.makedirs("embeddings", exist_ok=True)
    with open("embeddings/asv_embd_%s.pk" % (set_name), "wb") as f:
        pk.dump(asv_emb_dic, f)


def save_models(set_name, asv_embd_ext, device):
    utt2spk = {}
    utt_list = []

    for trn in SET_TRN[set_name]:
        meta_lines = open(trn, "r").readlines()

        for line in meta_lines:
            tmp = line.strip().split(" ")

            spk = tmp[0]
            utts = tmp[1].split(",")

            for utt in utts:
                if utt in utt2spk:
                    print("Duplicated utt error", utt)

                utt2spk[utt] = spk
                utt_list.append(utt)

    base_dir = SET_DIR[set_name]
    dataset = Dataset_ASVspoof2019_devNeval(utt_list, Path(base_dir))
    loader = DataLoader(
        dataset, batch_size=30, shuffle=False, drop_last=False, pin_memory=True
    )
    asv_emb_dic = {}

    print("Getting embedgins from set %s..." % (set_name))

    for batch_x, key in tqdm(loader):
        batch_x = batch_x.to(device)
        with torch.no_grad():
            batch_asv_emb = asv_embd_ext(batch_x, aug=False).detach().cpu().numpy()

        for k, asv_emb in zip(key, batch_asv_emb):
            utt = k
            spk = utt2spk[utt]
            
            if spk not in asv_emb_dic:
                asv_emb_dic[spk] = []

            asv_emb_dic[spk].append(asv_emb)

    for spk in asv_emb_dic:
        asv_emb_dic[spk] = np.mean(asv_emb_dic[spk], axis=0)

    with open("embeddings/spk_model.pk_%s" % (set_name), "wb") as f:
        pk.dump(asv_emb_dic, f)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ecapa_weight", type=str, default="./ECAPATDNN/exps/pretrain.model"
    )

    return parser.parse_args()


def main():
    args = get_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))

    asv_embd_ext = ECAPA_TDNN(C=1024)
    load_parameters(asv_embd_ext.state_dict(), args.ecapa_weight)
    asv_embd_ext.to(device)
    asv_embd_ext.eval()

    for set_name in SET_PARTITION:
        save_embeddings(
            set_name,
            asv_embd_ext,
            device,
        )
        if set_name == "trn":
            continue
        save_models(set_name, asv_embd_ext, device)


if __name__ == "__main__":
    main()
