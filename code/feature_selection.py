import anndata as ad
from glob import glob
import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA
import scipy
import statsmodels.api as sm
import scipy.stats as ss
from scipy.interpolate import interp1d
import multiprocessing as mp
from sklearn.feature_extraction.text import TfidfTransformer


# Perform Signac TF-IDF (count_mat: peak*cell)
def tfidf2(count_mat): 
    tf_mat = 1.0 * count_mat / np.tile(np.sum(count_mat,axis=0), (count_mat.shape[0],1))
    signac_mat = np.log(1 + np.multiply(1e4*tf_mat,  np.tile((1.0 * count_mat.shape[1] / np.sum(count_mat,axis=1)).reshape(-1,1), (1,count_mat.shape[1]))))
#     return scipy.sparse.csr_matrix(signac_mat)
    return signac_mat

def tfidf1(count_mat): 
    nfreqs = 1.0 * count_mat / np.tile(np.sum(count_mat,axis=0), (count_mat.shape[0],1))
    tfidf_mat = np.multiply(nfreqs, np.tile(np.log(1 + 1.0 * count_mat.shape[1] / np.sum(count_mat,axis=1)).reshape(-1,1), (1,count_mat.shape[1])))
    return tfidf_mat

def tfidf3(count_mat): 
    model = TfidfTransformer(smooth_idf=False, norm="l2")
    model = model.fit(np.transpose(count_mat))
    model.idf_ -= 1
    tf_idf = np.transpose(model.transform(np.transpose(count_mat)))
    return tf_idf.todense()

def pearson_sparse(count, nY):
    '''
    count以np.array的形式存储
    '''
    Y_mean = nY / count.shape[0]
    # mean = []
    # var = []
    mean = np.zeros(count.shape[1])
    var = np.zeros(count.shape[1])
    count_new = count - Y_mean
    b = np.sum(count_new**2,axis=0)
    for i in range(count.shape[1]):
        if i % 10000 == 0:
            print(f"processing...{i}/{count.shape[1]} {int(i/count.shape[1] * 100)}%")
        # 计算第i个peak与其他所有的相关系数，得到均值和方差
        X = count[:,i]
        X_new = count_new[:,i]
        a = np.dot(X_new,count_new)
        corr = a / (np.sqrt(np.multiply(np.dot(X_new, X_new),b)))
        
        # mean.append(np.mean(corr))
        # var.append(np.mean(corr**2))
        mean[i] = np.mean(corr)
        var[i] = np.mean(corr**2)
    print(f"processing...{count.shape[1]}/{count.shape[1]} {int((count.shape[1])/count.shape[1] * 100)}%")
    # return np.array(mean),np.array(var)
    return mean, var

def cos_sparse(count):
    '''
    count以np.array的形式存储
    '''
    mean = []
    var = []
    y = count.T
    yy = np.sum(y ** 2, axis=1) ** 0.5
    y = y / yy[:, np.newaxis]
    for i in range(0, count.shape[1], 500):
        if i % 10000 == 0:
            print(f"processing...{i}/{count.shape[1]} {int(i/count.shape[1] * 100)}%")
        # 计算第i个peak与其他所有的相关系数，得到均值和方差
        x = count[:,i:i+500].T
        xx = np.sum(x ** 2, axis=1) ** 0.5
        x = x / xx[:, np.newaxis]
        corr = 1 - np.dot(x, y.transpose())

        mean += list(np.mean(corr,axis=1))
        var += list(np.mean(corr**2,axis=1))
    print(f"processing...{count.shape[1]}/{count.shape[1]} {int((count.shape[1])/count.shape[1] * 100)}%")
    return np.array(mean),np.array(var)

def feature_selection(anndata, select_num, seed_base, filter_para, tfidf="tfidf2", corr="pearson"):
    print(anndata)

    Y = np.array(anndata.X.todense()>0,dtype = 'float32')
    # Y = np.array(ATAC_all.X>0,dtype = 'float32')
    Y = scipy.sparse.csc_matrix(Y)
    print('预选在多于{}%的细胞中开放的peak'.format(filter_para*100))
    peak_sum = np.sum(Y, axis=0)
    peak_sum = np.array(peak_sum).reshape(-1)
    idx = peak_sum > anndata.n_obs * filter_para
    ATAC_object = anndata[:, idx]
    num1 = ATAC_object.n_vars
    print(ATAC_object)

    if tfidf == "tfidf1":
        ATAC_count = tfidf1(ATAC_object.X.todense().T)
    elif tfidf == "tfidf2":
        ATAC_count = tfidf2(ATAC_object.X.todense().T)
    else:
        ATAC_count = tfidf3(ATAC_object.X.todense().T)
    count = np.array(ATAC_count)
    count = PCA(n_components=100,random_state=int(seed_base*1000)).fit_transform(count)
    print(count.shape)
    count = count.T
    nY = np.sum(count, axis=0)

    if corr == "pearson":
        peak_mean, peak_var = pearson_sparse(count, nY)
    elif corr == "cosine":
        peak_mean, peak_var = cos_sparse(count)
    else:
        order_count = np.zeros(count.shape)
        for i in range(count.shape[1]):
            order_count[:,i] = np.argsort(count[:,i])
        nY = np.sum(order_count, axis=0)
        peak_mean, peak_var = pearson_sparse(order_count, nY)
    # 去除对角线影响
    peak_mean = (peak_mean * ATAC_count.shape[1] - 1) / (ATAC_count.shape[1] - 1)
    peak_var = (peak_var * ATAC_count.shape[1] - 1) / (ATAC_count.shape[1] - 1)

    frac = 0.2
    # 第一次拟合
    print('第一次拟合')
    sort_idx = np.argsort(peak_mean)
    peak_mean_lowess = peak_mean[sort_idx]
    peak_var_lowess = peak_var[sort_idx]
    lowess = sm.nonparametric.lowess
    yest = lowess(exog=peak_mean_lowess, endog=peak_var_lowess, frac=frac, is_sorted=True)[:,1]

    res = yest - peak_var_lowess
    mu = np.mean(res)
    std = np.std(res)
    pvalue = ss.norm(loc=mu, scale=std).cdf(res)
    idx_remove = np.bitwise_or(pvalue < 0.05, pvalue > 0.95)
    idx_reserve = np.bitwise_and(pvalue > 0.05, pvalue < 0.95)

    peak_mean_lowess = peak_mean[sort_idx][idx_reserve]
    peak_var_lowess = peak_var[sort_idx][idx_reserve]

    # 第二次拟合
    print('第二次拟合')
    lowess = sm.nonparametric.lowess
    yest = lowess(exog=peak_mean_lowess, endog=peak_var_lowess, frac=frac, is_sorted=True)[:,1]

    f = interp1d(peak_mean_lowess, yest, bounds_error=False) # 插值
    yest_all = f(peak_mean[sort_idx])

    res = yest_all - peak_var[sort_idx]
    res_notnan = np.delete(res, np.where(np.isnan(res)))
    res[np.where(np.isnan(res))] = np.min(res_notnan)
    # 计算res在高斯分布中的概率
    mu = np.mean(res_notnan)
    std = np.std(res_notnan)

    pvalue = ss.norm(loc=mu, scale=std).cdf(res)
    idx_remove = np.bitwise_or(pvalue < 0.05, pvalue > 0.95)
    idx_reserve = np.bitwise_and(pvalue > 0.05, pvalue < 0.95)

    peak_mean_lowess = peak_mean[sort_idx][idx_reserve]
    peak_var_lowess = peak_var[sort_idx][idx_reserve]

    # 第三次拟合
    print('第三次拟合')
    lowess = sm.nonparametric.lowess
    yest = lowess(exog=peak_mean_lowess, endog=peak_var_lowess, frac=frac, is_sorted=True)[:,1]

    f = interp1d(peak_mean_lowess, yest, bounds_error=False) # 插值
    yest_all = f(peak_mean[sort_idx])

    # 选择peak
    res = yest_all - peak_var[sort_idx]
    res_notnan = np.delete(res, np.where(np.isnan(res)))
    res[np.where(np.isnan(res))] = np.min(res_notnan)
    # 计算res在高斯分布中的概率
    mu = np.mean(res_notnan)
    std = np.std(res_notnan)
    pvalue = ss.norm(loc=mu, scale=std).cdf(res)

    ATAC_count = tfidf2(ATAC_object.X.todense().T).T
    if select_num > num1:
        print('The number of features to be selected is greater than the total number of features...')
        select_num = num1

    res_sort_idx = np.argsort(np.abs(res))
    res_sort_idx = res_sort_idx[-select_num:]

    peak_mean_lowess = peak_mean[sort_idx]
    peak_var_lowess = peak_var[sort_idx]
    idx = sort_idx[res_sort_idx]

    label = np.array(['reserve']*peak_mean_lowess.shape[0])
    label[idx] = 'remove'

    res_select = res[res_sort_idx]
    res_select_mean, res_select_var = np.mean(np.abs(res_select)), np.var(res_select)

    ATAC_count_filter = ATAC_count[:,idx]
    
    return(idx, ATAC_count_filter)