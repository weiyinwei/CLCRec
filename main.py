import argparse
import os
import time
import numpy as np
import torch
import random
from Dataset import TrainingDataset, data_load
from model_CLCRec import CLCRec
from torch.utils.data import DataLoader
from Train import train
from Full_rank import full_ranking
from torch.utils.tensorboard import SummaryWriter
###############################248###########################################

def init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='Seed init.')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--data_path', default='movielens', help='Dataset path')
    parser.add_argument('--save_file', default='', help='Filename')

    parser.add_argument('--PATH_weight_load', default=None, help='Loading weight filename.')
    parser.add_argument('--PATH_weight_save', default=None, help='Writing weight filename.')
    parser.add_argument('--prefix', default='', help='Prefix of save_file.')

    parser.add_argument('--l_r', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--lr_lambda', type=float, default=1, help='Weight loss one.')
    parser.add_argument('--reg_weight', type=float, default=1e-1, help='Weight decay.')
    parser.add_argument('--temp_value', type=float, default=1, help='Contrastive temp_value.')
    parser.add_argument('--model_name', default='SSL', help='Model Name.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--num_neg', type=int, default=512, help='Negative size.')
    parser.add_argument('--num_epoch', type=int, default=1000, help='Epoch number.')
    parser.add_argument('--num_workers', type=int, default=1, help='Workers number.')
    parser.add_argument('--num_sample', type=float, default=0.5, help='Workers number.')

    parser.add_argument('--dim_E', type=int, default=64, help='Embedding dimension.')
    parser.add_argument('--topK', type=int, default=10, help='Workers number.')
    parser.add_argument('--step', type=int, default=2000, help='Workers number.')
    
    parser.add_argument('--has_v', default='False', help='Has Visual Features.')
    parser.add_argument('--has_a', default='False', help='Has Acoustic Features.')
    parser.add_argument('--has_t', default='False', help='Has Textual Features.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = init()
    
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    ##########################################################################################################################################
    data_path = args.data_path
    save_file_name = args.save_file

    learning_rate = args.l_r
    lr_lambda = args.lr_lambda
    reg_weight = args.reg_weight
    batch_size = args.batch_size
    num_workers = args.num_workers
    num_epoch = args.num_epoch
    num_neg = args.num_neg
    num_sample = args.num_sample
    topK = args.topK
    prefix = args.prefix
    model_name = args.model_name
    temp_value = args.temp_value
    step = args.step

    has_v = True if args.has_v == 'True' else False
    has_a = True if args.has_a == 'True' else False
    has_t = True if args.has_t == 'True' else False

    dim_E = args.dim_E
    is_word = True if data_path == 'tiktok' else False
    writer = SummaryWriter()
    # with open(data_path+'/result/result{0}_{1}.txt'.format(l_r, reg_weight), 'w') as save_file:
    #     save_file.write('---------------------------------lr: {0} \t reg_weight:{1} ---------------------------------\r\n'.format(l_r, reg_weight))
    ##########################################################################################################################################
    print('Data loading ...')

    num_user, num_item, num_warm_item, train_data, val_data, val_warm_data, val_cold_data, test_data, test_warm_data, test_cold_data, v_feat, a_feat, t_feat = data_load(data_path)
    
    dir_str = './Data/' + data_path
    user_item_all_dict = np.load(dir_str+'/user_item_all_dict.npy', allow_pickle=True).item()
    user_item_train_dict = np.load(dir_str+'/user_item_train_dict.npy', allow_pickle=True).item()

    warm_item = torch.tensor(np.load(dir_str + '/warm_set.npy'))
    cold_item = torch.tensor(np.load(dir_str + '/cold_set.npy'))

    train_dataset = TrainingDataset(num_user, num_item, user_item_all_dict, data_path, train_data, num_neg)
    
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)

    print('Data has been loaded.')
    ##########################################################################################################################################
    model = CLCRec(num_user, num_item, num_warm_item, train_data, reg_weight, dim_E, v_feat, a_feat, t_feat, temp_value, num_neg, lr_lambda, is_word, num_sample).cuda()
    
    ##########################################################################################################################################
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': learning_rate}])#, 'weight_decay': reg_weight}])
    ##########################################################################################################################################
    max_precision = 0.0
    max_recall = 0.0
    max_NDCG = 0.0
    num_decreases = 0 
    max_val_result = max_val_result_warm = max_val_result_cold = max_test_result = max_test_result_warm = max_test_result_cold = list()
    for epoch in range(num_epoch):
        loss, mat = train(epoch, len(train_dataset), train_dataloader, model, optimizer, batch_size, writer)

        if torch.isnan(loss):
            print(model.result)
            with open('./Data/'+data_path+'/result_{0}.txt'.format(save_file_name), 'a') as save_file:
                    save_file.write('lr:{0} \t reg_weight:{1} is Nan\r\n'.format( learning_rate, reg_weight))
            break
        torch.cuda.empty_cache()

        # train_precision, train_recall, train_ndcg = full_ranking(epoch, model, user_item_inter, user_item_inter, True, step, topK, 'Train', writer)
        val_result = full_ranking(epoch,  model, val_data, user_item_train_dict, None, False, step, topK, 'Val/', writer)
        
        val_result_warm = full_ranking(epoch,  model, val_warm_data, user_item_train_dict, cold_item, False, step, topK, 'Val/warm_', writer)
        
        val_result_cold = full_ranking(epoch,  model, val_cold_data, user_item_train_dict, warm_item, False, step, topK, 'Val/cold_', writer)
        
        test_result = full_ranking(epoch, model, test_data, user_item_train_dict, None, False, step, topK, 'Test/', writer)
        
        test_result_warm = full_ranking(epoch, model, test_warm_data, user_item_train_dict, cold_item, False, step, topK, 'Test/warm_', writer)
        
        test_result_cold = full_ranking(epoch, model, test_cold_data, user_item_train_dict, warm_item, False, step, topK, 'Test/cold_', writer)


        if val_result[1] > max_recall:
            pre_id_embedding = model.id_embedding
            max_recall = val_result[1]
            max_val_result = val_result
            max_val_result_warm = val_result_warm
            max_val_result_cold = val_result_cold
            max_test_result = test_result
            max_test_result_warm = test_result_warm
            max_test_result_cold = test_result_cold
            num_decreases = 0
        else:
            if num_decreases > 5:
                with open('./Data/'+data_path+'/result_{0}.txt'.format(save_file_name), 'a') as save_file:
                    save_file.write(str(args))
                    save_file.write('\r\n-----------Val Precition:{0:.4f} Recall:{1:.4f} NDCG:{2:.4f}-----------'.format(max_val_result[0], max_val_result[1], max_val_result[2]))
                    save_file.write('\r\n-----------Val Warm Precition:{0:.4f} Recall:{1:.4f} NDCG:{2:.4f}-----------'.format(max_val_result_warm[0], max_val_result_warm[1], max_val_result_warm[2]))
                    save_file.write('\r\n-----------Val Cold Precition:{0:.4f} Recall:{1:.4f} NDCG:{2:.4f}-----------'.format(max_val_result_cold[0], max_val_result_cold[1], max_val_result_cold[2]))
                    save_file.write('\r\n-----------Test Precition:{0:.4f} Recall:{1:.4f} NDCG:{2:.4f}-----------'.format(max_test_result[0], max_test_result[1], max_test_result[2]))
                    save_file.write('\r\n-----------Test Warm Precition:{0:.4f} Recall:{1:.4f} NDCG:{2:.4f}-----------'.format(max_test_result_warm[0], max_test_result_warm[1], max_test_result_warm[2]))
                    save_file.write('\r\n-----------Test Cold Precition:{0:.4f} Recall:{1:.4f} NDCG:{2:.4f}-----------'.format(max_test_result_cold[0], max_test_result_cold[1], max_test_result_cold[2]))
                break
            else:
                num_decreases += 1