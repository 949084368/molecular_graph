# **Environment**

The most important python packages are:
- python == 3.6.7
- pytorch == 1.2.0
- torch == 0.4.1
- tensorboard == 1.13.1
- rdkit == 2019.09.3
- scikit-learn == 0.22.2.post1
- hyperopt == 0.2.5
- numpy == 1.18.2

For using our model more conveniently, we provide the environment file *<environment.txt>* to install environment directly.


---
# **Command**

### **1. Train**
Use train.py

Args:
  - data_path : The path of input CSV file. *E.g. input.csv*
  - dataset_type : The type of dataset. *E.g. classification  or  regression*
  - save_path : The path to save output model. *E.g. model_save*
  - log_path : The path to record and save the result of training. *E.g. log*

---
# **Data**

We provide the three public benchmark datasets used in our study: *<Data.rar>*

Or you can use your own dataset:
### 1. For training
The dataset file should be a **CSV** file with a header line and label columns. E.g.
```
SMILES,BT-20
O(C(=O)C(=O)NCC(OC)=O)C,0
FC1=CNC(=O)NC1=O,0
...


---
# **Example**
### 1. Training a model on BACE dataset
Decompress the Data.rar and find BACE dataset file in *Data/MoleculeNet/bace.csv*.

Use command:

`python train.py  --data_path Data/MoleculeNet/bace.csv  --dataset_type classification  --save_path model_save/bace  --log_path log/bace`
