import argparse
import scanpy as sc
import os
import numpy as np

from feature_selection import feature_selection


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--load_path', type=str,default=None, help='h5ad file storage path.')
    parser.add_argument('-t', '--TFIDF', type=str,default="tfidf2", help='TF-IDF implementation.')
    parser.add_argument('-p', '--PC', type=int,default=100, help='Dimension of cell-wise PCA.')
    parser.add_argument('-c', '--corr', type=str,default="pearson", help='Correlation coefficient calculation method.')
    parser.add_argument('-n', '--select_number', type=int,default=20000, help='Number of selected features.')
    parser.add_argument('-s', '--seed_base', type=int,default=2, help='Random seed.')
    opt = parser.parse_args()

    dataset = opt.load_path
    tfidf = opt.TFIDF
    PC_num = opt.PC
    corr = opt.corr
    select_num = opt.select_number
    seed_base = opt.seed_base

    ATAC_all = sc.read_h5ad(dataset)
   
    idx, ATAC_count_filter = feature_selection(ATAC_all, select_num, seed_base, 0.01, tfidf, corr, PC_num)

    if not os.path.exists('./result'):
        os.makedirs('./result')
    files = os.listdir('./result')
    files_num = len(files)

    np.savetxt('./result/filter_matrix_{}.txt'.format(files_num+1), ATAC_count_filter, delimiter=",")
    np.savetxt('./result/filter_idx_{}.txt'.format(files_num+1), idx, delimiter=",")