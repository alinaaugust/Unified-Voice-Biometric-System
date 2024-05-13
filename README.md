# Unified Voice Biometric System Against Spoofing Attacks

This project contains the implementation of some joint SASV solutions, adopted for the physical access scenario. It includes baselines from [here](https://arxiv.org/pdf/2203.14732); probabilistic framework-based solution from [here](https://www.isca-archive.org/pdfs/odyssey_2022/zhang22b_odyssey.pdf); sequential cascade described in [here](https://www.isca-archive.org/interspeech_2022/wang22ea_interspeech.pdf) and FiLM-based system from [here](https://www.isca-archive.org/interspeech_2022/choi22b_interspeech.pdf).

### Data Preparation

The ASV and CM embeddings can be downloaded [here](https://drive.google.com/file/d/14fAevLEh_AQvqggiTfrWFX2YJ2f2ekGh/view?usp=sharing). The extraction proccess is based on the [SASV22 Baseline repository](https://github.com/sasv-challenge/SASVC2022_Baseline) and can be found in [save_embeddings](/save_embeddings). Extracting ASV embeddings requires these [two](https://github.com/TaoRuijie/ECAPA-TDNN) [repositories](https://github.com/clovaai/aasist); in order to extract CM embeddings you will need [this](https://github.com/asvspoof-challenge/2021/tree/main/PA/Baseline-LFCC-LCNN) repository.

### System Description
We use the [ECAPA-TDNN](https://github.com/TaoRuijie/ECAPA-TDNN) as the ASV subsystem and the [LFCC-LCNN](https://github.com/asvspoof-challenge/2021/tree/main/PA/Baseline-LFCC-LCNN) as the CM subsystem. 

Detailed description of the systems, as well as all the information about the project, is available [in the paper](/paper/Avgustyonok_coursework_final).

The code is based on PyTorch. The main parameters you need to specify are the directory for the results of the run with `-o` and the name of the model with `-m`. Available models: `baseline1`, `pr_l_i`, `pr_s_i`, `baseline1_l_i`, `baseline1_s_i`, `casc_asv_cm`, `casc_cm_asv`, `film`. See the paper for more details.
```
python3 main_train.py -o ./exp_result/ -m casc_asv_cm
```

### Credits
This repository is based on the [official implementation](https://github.com/yzyouzhang/SASV_PR) of a paper "A Probabilistic Fusion Framework for Spoofing Aware Speaker Verification".

#### Fulfilled by:
Avgustyonok Alina Alekseevna \
Student of the Group БПМИ211 \
Faculty of Computer Science, HSE University 

#### Project Supervisor:
Grinberg Petr Markovich \
Visiting Lecturer \
Faculty of Computer Science, HSE University

