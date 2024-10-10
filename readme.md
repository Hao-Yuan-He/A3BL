This is the official code of paper *Ambiguity-Aware Abductive Learning*, In: Proceedings of 41st International Conference on Machine Learning.

## Set up 
```bash
# Require swi-prolog >= 8.0
# In ubuntu system  22.04 or higher
sudo apt install swi-prolog
# Otherwise, please download the specific version of swi-prolog and install it manually.
```

```bash
# Create Conda Enviroment
conda create --name abl python=3.10 
conda activate abl 

# Login your wandb account(if not, the logging process will encounter error.)
# The main results can be seen on the **wandb webpage, check it please!**
wandb login
```

## Experiment reproduce

### Digit Addition
```bash 
cd examples/addition

# dataset in [MNIST, KMNIST, SVHN, CIFAR]
# digit_size in [1, 2, 3, 4]
# Note: if digit_size is 2,3,4 the learning process will not be very quick, be patient.
python wsabl.py --dataset $dataset --digit_size $digit_size
```


### Handwritten Formula Recognition
```bash

# HWF
cd examples/hwf 
cd datasets
tar xf data.tgz 
cd ..
python wsabl.py 

# HWF-CIFAR
cd examples/hwf-cifar
cd datasets
tar xf data.tgz 
cd ..
python wsabl.py

# HWF-SVHN
cd examples/hwf-svhn
cd datasets
tar xf data.tgz 
cd ..
python wsabl.py
```




## AcKnowledgement
Thanks for the great libs: 
- ABLKit: https://github.com/AbductiveLearning/ABLkit (The code is mainly based on a previous commit of ABLKit, although there should be a litter different between APIs and details.)
- Weights & Bias: https://wandb.ai/site  (for logging)

## Others
The name `wsabl` is a short for `Weakly Supervised Abductive Learning`, which is a previous and initial naming way of A3BL.
Any question or suggestion, please contact me: `alkane0050@gmail.com`(prefered).

## Licecense
This project is licensed under the terms of the MIT license.