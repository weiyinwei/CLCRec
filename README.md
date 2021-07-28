# Contrastive Learning for Cold-start Recommendation

This is our Pytorch implementation for the paper:  
> Yinwei Wei, Xiang Wang, Qi Li, Liqiang Nie, Yan Li, Xuanping Li, and Tat-Seng Chua (2021). Contrastive Learning for Cold-start Recommendation, Paper in ACM DL or [Paper in arXiv](https://arxiv.org/abs/2107.05315). In ACM MM`21, Chengdu, China, Oct. 20-24, 2021  
Author: Dr. Yinwei Wei (weiyinwei at hotmail.com)

## Introduction
In this work, we reformulate the cold-start item representation learning from an information-theoretic standpoint. It aims to maximize the mutual dependencies between item content and collaborative signals. Specifically, the representation learning is theoretically lower-bounded by the integration of two terms: mutual information between collaborative embeddings of users and items, and mutual information between collaborative embeddings and feature representations of items. To model such a learning process, we devise a new objective function founded upon contrastive learning and develop a new Contrastive Learning-based Cold-start Recommendation framework (CLCRec).

## Citation
If you want to use our codes and datasets in your research, please cite:

``` 
@inproceedings{CLCRec,
  title     = {Contrastive Learning for Cold-start Recommendation},
  author    = {Wei, Yinwei and 
               Wang, Xiang and 
               Qi, Li and
               Nie, Liqiang and 
               Li, Yan and 
               Li, Xuanqing and 
               Chua, Tat-Seng},
  booktitle = {Proceedings of the 29th ACM International Conference on Multimedia},
  pages     = {--},
  year      = {2021}
}
``` 


## Environment Requirement
The code has been tested running under Python 3.5.2. The required packages are as follows:
- Pytorch == 1.1.0
- torch-cluster == 1.4.2
- torch-geometric == 1.2.1
- torch-scatter == 1.2.0
- torch-sparse == 0.4.0
- numpy == 1.16.0

## Example to Run the Codes
The instruction of commands has been clearly stated in the codes.

- Movielens dataset  
`python main.py --model_name='CLCRec' --l_r=0.001 --reg_weight=0.1 --num_workers=4 --num_neg=128 --has_a=True --has_t=True --has_v=True --lr_lambda=0.5 --temp_value=2.0 --num_sample=0.5` 

- Amazon dataset  
`python main.py --model_name='CLCRec' --data_path=amazon --l_r=0.001 --reg_weight=0.001 --num_workers=4 --num_neg=512 --has_v=True --lr_lambda=0.9 --num_sample=0.5`  

Some important arguments:  


- `lr_lambda`: 
  It specifics the value of lambda to balance the U-I and R-E mutual information.

- `num_neg` 
  This parameter indicates the number of negative sampling.  
  
- `num_sample`:
  This parameter indicates the probability of hybrid contrastive training.
  
- `temp_value`:
   It specifics the temprature value in density ratio functions.
## Dataset
We provide two processed datasets: Movielens and Amazon. (The details could be found in our article)
For Kwai and Tiktok datasets, due to the copyright, please connect the owners of datasets.
