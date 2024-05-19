import torch
from torch import nn, resolve_neg
import numpy as np


class Baseline1(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "Baseline1"
        self.fc_cm = nn.Linear(160, 2)
        self.coss = nn.CosineSimilarity(dim=1, eps=1e-8)

    def forward_SV_prob(self, embd_asv_enr, embd_asv_tst):
        asv_cos = self.coss(embd_asv_enr, embd_asv_tst).unsqueeze(1)
        asv_score = asv_cos
        p_sv = asv_score
        return p_sv

    def forward_CM_prob(self, embd_cm_tst):
        cm_score = embd_cm_tst
        return cm_score

    def forward(self, embd_asv_enr, embd_asv_tst, embd_cm_tst):
        p_sv = self.forward_SV_prob(embd_asv_enr, embd_asv_tst)
        p_cm = self.forward_CM_prob(embd_cm_tst)
        x = p_sv + p_cm
        return x

    def calc_loss(self, embd_asv_enr, embd_asv_tst, embd_cm_tst, labels):
        return 0


def cos2prob_lin(cos):
    return (cos + 1) / 2


class Parallel_PR(nn.Module):
    def __init__(self, trainable=True, calibrator=None, map_function=None):
        super().__init__()
        assert (not calibrator) or (not map_function)
        self.name = "ProductRule"
        self.trainable = trainable
        self.calibrator = calibrator
        self.map_func = map_function
        self.fc_cm = nn.Linear(160, 1)
        self.coss = nn.CosineSimilarity(dim=1, eps=1e-8)
        self.sigmoid = nn.Sigmoid()
        if self.trainable:
            self.loss_sasv = nn.BCELoss(weight=torch.FloatTensor([0.9]))

    def forward_SV_prob(self, embd_asv_enr, embd_asv_tst):
        asv_cos = self.coss(embd_asv_enr, embd_asv_tst).unsqueeze(1)
        if self.calibrator:
            asv_cos = self.coss(embd_asv_enr, embd_asv_tst).unsqueeze(1)
            p_sv = self.calibrator.predict_proba(asv_cos.cpu().numpy())[:, 1]
            p_sv = torch.from_numpy(p_sv).to(embd_asv_enr.device).unsqueeze(1).float()
            return p_sv
        if self.map_func == "linear":
            return cos2prob_lin(asv_cos)
        elif self.map_func == "sigmoid":
            return self.sigmoid(asv_cos)
        else:
            raise ValueError("which function to calculate probability")

    def forward_CM_prob(self, embd_cm_tst):
        cm_score = embd_cm_tst
        p_cm = self.sigmoid(cm_score)
        return p_cm

    def forward(self, embd_asv_enr, embd_asv_tst, embd_cm_tst):
        p_sv = self.forward_SV_prob(embd_asv_enr, embd_asv_tst)
        p_cm = self.forward_CM_prob(embd_cm_tst)
        x = p_sv * p_cm
        return x

    def calc_loss(self, embd_asv_enr, embd_asv_tst, embd_cm_tst, labels):
        if not self.trainable:
            return 0
        sasv_score = self.forward(embd_asv_enr, embd_asv_tst, embd_cm_tst)
        loss = self.loss_sasv(sasv_score, labels.unsqueeze(1).float())
        return loss


class Baseline1_improved(nn.Module):
    def __init__(self, map_function="sigmoid"):
        super().__init__()
        self.name = "Baseline1+scaling"
        self.map_func = map_function
        self.fc_cm = nn.Linear(160, 1)
        self.coss = nn.CosineSimilarity(dim=1, eps=1e-8)
        self.sigmoid = nn.Sigmoid()

    def forward_SV_prob(self, embd_asv_enr, embd_asv_tst):
        asv_cos = self.coss(embd_asv_enr, embd_asv_tst).unsqueeze(1)
        if self.map_func == "linear":
            return cos2prob_lin(asv_cos)
        elif self.map_func == "sigmoid":
            return self.sigmoid(asv_cos)
        else:
            raise ValueError("which function to calculate probability")

    def forward_CM_prob(self, embd_cm_tst):
        cm_score = embd_cm_tst
        p_cm = self.sigmoid(cm_score)
        return p_cm

    def forward(self, embd_asv_enr, embd_asv_tst, embd_cm_tst):
        p_sv = self.forward_SV_prob(embd_asv_enr, embd_asv_tst)
        p_cm = self.forward_CM_prob(embd_cm_tst)
        x = p_sv + p_cm
        return x

    def calc_loss(self, embd_asv_enr, embd_asv_tst, embd_cm_tst, labels):
        return 0

class Cascade(nn.Module):
    def __init__(self, first="asv", threshold=None):
        super().__init__()
        self.name = "Cascade_" + first
        self.first = first
        self.min_cm = 0
        self.min_asv = 0
        self.threshold = threshold
        self.coss = nn.CosineSimilarity(dim=1, eps=1e-8)
        self.sigmoid = nn.Sigmoid()

    def forward_SV_prob(self, embd_asv_enr, embd_asv_tst):
        asv_cos = self.coss(embd_asv_enr, embd_asv_tst).unsqueeze(1)
        return self.sigmoid(asv_cos)
    
    def forward_CM_prob(self, embd_cm_tst):
        cm_score = embd_cm_tst
        p_cm = self.sigmoid(cm_score)
        return p_cm

    def forward(self, embd_asv_enr, embd_asv_tst, embd_cm_tst):
        p_sv = self.forward_SV_prob(embd_asv_enr, embd_asv_tst)
        p_cm = self.forward_CM_prob(embd_cm_tst)
        x = p_sv if self.first == 'asv' else p_cm
        for i, ans in enumerate(x):
            if ans < self.threshold:
                x[i] = self.min_cm if self.first == 'asv' else self.min_asv
            else:
                x[i] = p_cm[i] if self.first == 'asv' else p_sv[i]
        return x

    def calc_loss(self, embd_asv_enr, embd_asv_tst, embd_cm_tst, labels):
        return 0


class Film(nn.Module):
    def __init__(self, batch_size, trainable=True, asv_embd = 192, cm_embd=96):
        super().__init__()
        self.name = "FiLM"
        self.trainable = trainable
        self.sv_emb = asv_embd
        self.cm_emb = cm_embd
        self.bs = batch_size
        # define all for film part 
        self.sv_ln = nn.LayerNorm(self.sv_emb)
        self.cm_ln = nn.LayerNorm(self.cm_emb)
        self.film = nn.Sequential(
            nn.Linear(self.cm_emb, 2 * self.sv_emb),
            nn.ReLU(),
            nn.BatchNorm1d(2 * self.sv_emb)
        )
        self.gamma = torch.zeros(self.sv_emb)
        self.beta = torch.zeros(self.sv_emb)
        # define all for e_CM
        self.get_probs = nn.Sequential(
            nn.Linear(self.cm_emb, 2),
            nn.Softmax()
        )
        # define all for e_SV
        self.e_mod = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.sv_emb, self.sv_emb),
            nn.ReLU(),
            nn.Linear(self.sv_emb, self.sv_emb)
        )

        self.coss = nn.CosineSimilarity(dim=1, eps=1e-8)
        self.sigmoid = nn.Sigmoid()
        self.loss_sasv = nn.BCELoss(weight=torch.FloatTensor([0.9]))

    def forward(self, embd_asv_enr, embd_asv_tst, embd_cm_tst):
        film_params = self.film(self.cm_ln(embd_cm_tst))
        self.gamma, self.beta = film_params[:,:self.sv_emb], film_params[:,self.sv_emb:]
        e_mod1 = self.gamma * self.sv_ln(embd_asv_tst) + self.beta
        e_mod2 = self.e_mod(e_mod1)
        probs = self.get_probs(embd_cm_tst)
        x = torch.diag(probs[:, 0]) @ embd_asv_tst + torch.diag(probs[:, 1]) @ e_mod2
        res = self.coss(x, embd_asv_enr).reshape(-1, 1)
        return self.sigmoid(res)

    def calc_loss(self, embd_asv_enr, embd_asv_tst, embd_cm_tst, labels):
        if not self.trainable:
            return 0
        sasv_score = self.forward(embd_asv_enr, embd_asv_tst, embd_cm_tst)
        loss = self.loss_sasv(sasv_score, labels.unsqueeze(1).float())
        return loss
        
