# Cofea

Single-cell sequencing technologies provide a revolutionary understanding of cellular heterogeneity at the single-cell resolution. However, the high-dimensional and high-noise nature of single-cell data presents challenges for downstream analyses, and feature selection has thus become a critical step in processing and analyzing single-cell data. Feature selection approaches are mature for single-cell RNA sequencing (scRNA-seq) data, but there is a significant gap in the availability of effective methods for feature selection in single-cell chromatin accessibility (scCAS) data. Here, we present Cofea, a correlation-based framework, for selecting informative features by exploring the interaction between features derived from scCAS data.

![image](https://github.com/likeyi19/Cofea/blob/master/inst/Model.png)

## Installation  

```  
Package installation:
$ git clone https://github.com/likeyi19/Cofea   
$ cd Cofea   
$ pip install -r requirements.txt
```

## Tutorial

### demo

We provide a [quick-start notebook](https://github.com/likeyi19/Cofea/blob/master/code/demo.ipynb) notebook which describes the fundamentals in detail and reproduces the results of Cofea.

### cofea

Six parameters are required, including the path of dataset, TFIDF implementation method, number of PCS, correlation coefficient calculation method, number of selected features, and random seed. TFIDF is 'tfidf2' by default, the number of PCS is 100 by default, and the correlation coefficient is 'PCC' by default.

For exsample:
```
$ cd code/
$ python cofea.py -l ../data/scanpy.h5ad  -n 20000 -s 2
$ cd ..
```

Or you can get help in this way:
```  
$ python code/cofean.py -h
usage: cofea.py [-h] 
                [-l LOAD_PATH] 
                [-t TFIDF] 
                [-p PC] 
                [-c CORR] 
                [-n SELECT_NUMBER] 
                [-s SEED_BASE]

optional arguments:
  -h, --help            show this help message and exit
  -l LOAD_PATH, --load_path LOAD_PATH
  -t TFIDF, --TFIDF TFIDF
  -p PC, --PC PC
  -c CORR, --corr CORR
  -n SELECT_NUMBER, --select_number SELECT_NUMBER
  -s SEED_BASE, --seed_base SEED_BASE
'''

### Simulation

The overlapped proportion between the identified features and the ground-truth informative features offers a  intuitive and accurate metric than evaluating the performance based on downstream analysis tasks. To this end, we generated five high-fidelity scCAS datasets to enable a quantitative evaluation of feature selection methods. 

Taking data set S4 as an example, we draw the visualization and UMAP of the dataset, and compare the overlapped proportion between the features selected by each method and the cell type-specific features.

![image](https://github.com/likeyi19/Cofea/blob/master/inst/S4.png)

We provide a [notebook](https://github.com/likeyi19/Cofea/blob/master/code/simulation.ipynb) to reproduce the generation and testing process for simulation datasets

### Analysis

From the perspective of revealing biological insights, we evaluated Cofea via several numeric experiments including cell type-specific peaks annotation and candidate enhancers identification.

We provide a [notebook](https://github.com/likeyi19/Cofea/blob/master/code/analysis.ipynb) to analyze the sample dataset.

## Contact 
If you have any questions, you can contact me from the email: <931818472@qq.com>
