# SRFANet
## 1. For University-1652 dataset.
Train: run *train_university.py*, with --only_test = False.
Test: run *train_university.py*, with --only_test = True, and choose the model in --ckpt_path.

## 2. For SUES-200 dataset.
You need to split the origin dataset into the appropriate format using the script "SRFANet-->sample4geo-->dataset-->SUES-200-->split_datasets.py".
