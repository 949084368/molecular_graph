from argparse import Namespace
from logging import Logger
import numpy as np
import os
from gnn.train import fold_train
from gnn.tool import set_log, set_train_argument, get_task_name, mkdir

def training(args,log):
    info = log.info

    seed_first = args.seed
    data_path = args.data_path
    save_path = args.save_path

    score = []

    for num_fold in range(args.num_folds):
        info(f'Seed {args.seed}')
        args.seed = seed_first + num_fold
        args.save_path = os.path.join(save_path, f'Seed_{args.seed}')
        mkdir(args.save_path)

        fold_score = fold_train(args,log)

        score.append(fold_score)
    score = np.array(score)

    info(f'Running {args.num_folds} folds in total.')
    if args.num_folds > 1:
        for num_fold, fold_score in enumerate(score):
            info(f'Seed {seed_first + num_fold} : test {args.metric} = {np.nanmean(fold_score):.6f}')
            if args.task_num > 1:
                for one_name,one_score in zip(args.task_names,fold_score):
                    info(f'    Task {one_name} {args.metric} = {one_score:.6f}')
    ave_task_score = np.nanmean(score, axis=1)
    score_ave = np.nanmean(ave_task_score)
    score_std = np.nanstd(ave_task_score)
    info(f'Average test {args.metric} = {score_ave:.6f} +/- {score_std:.6f}')

    if args.task_num > 1:
        for i,one_name in enumerate(args.task_names):
            info(f'    average all-fold {one_name} {args.metric} = {np.nanmean(score[:, i]):.6f} +/- {np.nanstd(score[:, i]):.6f}')

    return score_ave, score_std

if __name__ == '__main__':
    args = Namespace(data_path='E:\\MoleculeFormer\\Data\\MoleculeNet\\bace.csv', save_path='model_save_test', log_path='log', dataset_type='classification',
                     is_multitask=0, task_num=1, split_type='ratio_random', split_ratio=[0.8, 0.1, 0.1], val_path=None,
                     test_path=None, seed=0, num_folds=4, metric='auc', epochs=20, batch_size=50, fp_type='mixed',
                     hidden_size=100, fp_2_dim=512, nhid=60, nheads=4, gat_scale=0.5, dropout=0.0, dropout_gat=0.0,
                     cuda=True, init_lr=0.0001, max_lr=0.001, final_lr=0.0001, warmup_epochs=2.0, num_lrs=1,model_type='GCNtransformer',noise_rate=0.0)

    # 注意调整 hidden_size
    log = set_log('train',args.log_path)
    training(args,log)

