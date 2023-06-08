from tool import *
import scipy
import umap

def evaluation(ATAC_all, ATAC_count_filter, seed_base):
    print('选择peak后矩阵维度为{}'.format(ATAC_count_filter.shape))
    ATAC_count_filter = PCA(n_components=10,random_state=int(seed_base*1000)).fit_transform(ATAC_count_filter)
    print('pca后矩阵维度为{}'.format(ATAC_count_filter.shape))
    ATAC_pca = sc.AnnData(scipy.sparse.csc_matrix(ATAC_count_filter),dtype = 'float32')
    ATAC_pca.obs['label'] = list(ATAC_all.obs['label'])

    # umap
    label = ATAC_pca.obs['label']
    proj = umap.UMAP().fit_transform(ATAC_count_filter)
    df = {'component_1':proj[:, 0],\
        'component_2':proj[:, 1], \
        'label':label}
    df = pd.DataFrame(df)
    ax = sns.scatterplot(x="component_1", y="component_2", hue="label",palette = 'Dark2', s=5,linewidth = 0.05, data=df)
    ax.legend()
    plt.show()

    sc.pp.neighbors(ATAC_pca, n_neighbors=15, use_rep='X',random_state=seed_base*1000)
    louvain_df = run_louvain(ATAC_pca,'label','cluster',seed=int(seed_base*1000))
    print("cluster results:")
    print(louvain_df)
    # 计算轮廓系数
    ATAC_pca.obsm['latent'] = scipy.sparse.csc_matrix(ATAC_count_filter)
    print("ASW:")
    print(silhouette(adata=ATAC_pca, group_key='label', embed='latent'))
    # 计算clisi_graph
    ATAC_pca.obsm['X_emb'] = scipy.sparse.csc_matrix(ATAC_count_filter)
    ATAC_pca.obs['label'] = ATAC_pca.obs['label'].astype('category')
    print("cLISI:")
    print(clisi_graph(adata=ATAC_pca, label_key='label', type_='embed'))