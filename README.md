# Periodic Source Detection in Time Series



## An Unsupervised Approach for Periodic Source Detection in Time Series (ICML 2024, Official Code)

Berken Utku Demirel, Christian Holz<br/>

<p align="center">
</p>

---

> Detection of periodic patterns of interest within noisy time series data plays a critical role in various tasks, spanning from health monitoring to behavior analysis.
Existing learning techniques often rely on labels or clean versions of signals for detecting the periodicity, and those employing self-supervised learning methods are required to apply proper augmentations, which is already challenging for time series and can result in collapse---all representations collapse to a single point due to strong augmentations.
In this work, we propose a novel method to detect the periodicity in time series without the need for any labels or requiring tailored positive or negative data generation mechanisms with specific augmentations.
We mitigate the collapse issue by ensuring the learned representations retain information from the original samples without imposing any random variance constraints on the batch.
Our experiments in three time series tasks against state-of-the-art learning methods show that the proposed approach consistently outperforms prior works, achieving performance improvements of more than 45--50%, showing its effectiveness.


Contents
----------

* [Datasets](#datasets)
* [Commands](#commands)
* [TLDR](#tldr)

Datasets
----------
1. Datasets
- `Heart rate prediction`  [IEEE SPC12 and IEEE SPC22](https://signalprocessingsociety.org/community-involvement/ieee-signal-processing-cup-2015), [DaLiA](https://archive.ics.uci.edu/dataset/495/ppg+dalia).
- `Respiratory rate`  [CapnoBase](https://borealisdata.ca/dataverse/capnobase).
- `Step counting`  [Clemson](https://sites.google.com/view/rmattfeld/pedometer-dataset).
2. After downloading the raw data, they should be processed with the corresponding [scripts](https://github.com/eth-siplab/Unsupervised_Periodicity_Detection/tree/main/Heuristic_and_data_prep), if there is any.

Commands
----------
The command to run the proposed approach:
```
python main.py --cuda 0 --dataset 'ieee_small'
```
The command to run the self-supervised learning methods:

```
python main_SSL_LE.py --framework 'barlowtwins' --backbone 'DCL' --n_epoch 120 --batch_size 256 --lr 3e-3 --lr_cls 0.03 --cuda 0 --dataset 'dalia' --data_type 'ppg' --cases 'subject_large_ssl_fn' --aug1 'perm_jit' --aug2 'perm_jit'
```

The command to run the supervised learning methods:
```
python main_supervised_baseline.py --backbone 'DCL' --dataset 'dalia' --data_type 'ppg' --cuda 0
```


TLDR
----------
We present a novel method (two regularizers) for detecting periodic patterns in time series data without using labels or specific augmentations. 
Our approach avoids common issues like representation collapse and outperforms existing methods across three different tasks.
The proposed regularizers can be found in [the training script](https://github.com/eth-siplab/Unsupervised_Periodicity_Detection/blob/main/trainer.py) under function name freq_losses. 
