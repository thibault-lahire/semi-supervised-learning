import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as sd
from scipy.io import loadmat
import os

from helper import build_similarity_graph, build_laplacian, plot_classification, label_noise, \
                    plot_classification_comparison, plot_clusters, plot_graph_matrix

#np.random.seed(50)


def build_laplacian_regularized(X, laplacian_regularization, var, eps, k, laplacian_normalization):
    """
    Function to construct a regularized Laplacian from data.

    :param X: (n x m) matrix of m-dimensional samples
    :param laplacian_regularization: regularization to add to the Laplacian (parameter gamma)
    :param var: the sigma value for the exponential function, already squared
    :param eps: threshold eps for epsilon graphs
    :param k: number of neighbours k for k-nn. If zero, use epsilon-graph
    :param laplacian_normalization: string selecting which version of the laplacian matrix to construct
                                    'unn':  unnormalized,
                                    'sym': symmetric normalization
                                    'rw':  random-walk normalization
    :return: Q (n x n ) matrix, the regularized Laplacian
    """
    # build the similarity graph W
    W = build_similarity_graph(X, var, eps, k)

    """
    Build the Laplacian L and the regularized Laplacian Q.
    Both are (n x n) matrices.
    """
    L = build_laplacian(W, laplacian_normalization)

    # compute Q
    Q = L + laplacian_regularization*np.eye(W.shape[0])

    return Q




def mask_labels(Y, l):
    """
    Function to select a subset of labels and mask the rest.

    :param Y:  (n x 1) label vector, where entries Y_i take a value in [1, ..., C] , where C is the number of classes
    :param l:  number of unmasked (revealed) labels to include in the output
    :return:  Y_masked:
               (n x 1) masked label vector, where entries Y_i take a value in [1, ..., C]
               if the node is labeled, or 0 if the node is unlabeled (masked)
    """
    num_samples = np.size(Y, 0)

    """
     randomly sample l nodes to remain labeled, mask the others   
    """
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    indices = indices[:l]

    Y_masked = np.zeros(num_samples)
    Y_masked[indices] = Y[indices]

    return Y_masked





def hard_hfs(X, Y, laplacian_regularization, var, eps, k, laplacian_normalization):
    """
    TO BE COMPLETED

    Function to perform hard (constrained) HFS.

    :param X: (n x m) matrix of m-dimensional samples
    :param Y: (n x 1) vector with nodes labels [0, 1, ... , num_classes] (0 is unlabeled)
    :param laplacian_regularization: regularization to add to the Laplacian
    :param var: the sigma value for the exponential function, already squared
    :param eps: threshold eps for epsilon graphs
    :param k: number of neighbours k for k-nn. If zero, use epsilon-graph
    :param laplacian_normalization: string selecting which version of the laplacian matrix to construct
                                    'unn':  unnormalized,
                                    'sym': symmetric normalization
                                    'rw':  random-walk normalization
    :return: labels, class assignments for each of the n nodes
    """
    Cl = np.unique(Y)
    num_classes = len(Cl)-1

    """
    Build the vectors:
    l_idx = (l x num_classes) vector with indices of labeled nodes
    u_idx = (u x num_classes) vector with indices of unlabeled nodes
    """
    l_idx = []
    u_idx = []
    for i in range(len(Y)):
        if Y[i] == 0:
            u_idx.append(i)
        else:
            l_idx.append(i)
    
    l_idx = np.asarray(l_idx)     
    u_idx = np.asarray(u_idx)

    """
    Compute the hfs solution, remember that you can use the functions build_laplacian_regularized and 
    build_similarity_graph    
    
    f_l = (l x num_classes) hfs solution for labeled data. It is the one-hot encoding of Y for labeled nodes.   
    
    example:         
        if Cl=[0,3,5] and Y=[0,0,0,3,0,0,0,5,5], then f_l is a 3x2  binary matrix where the first column codes 
        the class '3'  and the second the class '5'.    
    
    In case of 2 classes, you can also use +-1 labels      
        
    f_u = array (u x num_classes) hfs solution for unlabeled data
    
    f = array of shape(num_samples, num_classes)
    """


    L=build_laplacian_regularized(X, laplacian_regularization,var, eps, k, laplacian_normalization)
    L_uu=L[:,u_idx][u_idx]
    L_ul=L[:,l_idx][u_idx]

    Y_l=Y[l_idx]
    f_l = np.zeros((len(l_idx),num_classes))
    for i in range(len(f_l)):
        for j in range(1,num_classes+1): # one-hot-encoding
            if Y_l[i] == j:
                f_l[i,j-1] = 1
                
    f_u=np.linalg.inv(L_uu).dot(-L_ul.dot(f_l))

    """
    compute the labels assignment from the hfs solution   
    labels: (n x 1) class assignments [1,2,...,num_classes]    
    """
    Y[u_idx]=np.argmax(f_u,axis=1)+1
    labels=Y
    return labels





def two_moons_hfs():
    """
    TO BE COMPLETED.

    HFS for two_moons data.
    """

    """
    Load the data. At home, try to use the larger dataset (question 1.2).    
    """
    # load the data
    in_data = loadmat(os.path.join('data', 'data_2moons_hfs.mat'))
    #in_data = loadmat(os.path.join('data', 'data_2moons_hfs_large.mat'))
    X = in_data['X']
    Y = in_data['Y'].squeeze()


    """
    Choose the experiment parameters
    """
    var = 1.0
    eps = 0
    k = 6
    laplacian_regularization = 0.000000001
    laplacian_normalization = ''
    c_l = 0.96
    c_u = 0.04

    # number of labeled (unmasked) nodes provided to the hfs algorithm
    l = 4

    # mask labels
    Y_masked = mask_labels(Y, l)
    
    if np.size(np.where(Y_masked == 1)) == 4 or np.size(np.where(Y_masked == 2)) == 4:
        print("WARNING : the randomness was such that the four revealed labels are in the same class")

    """
    compute hfs solution using either soft_hfs or hard_hfs
    """
    #labels = hard_hfs(X, Y_masked, laplacian_regularization = laplacian_regularization, var = var, eps=eps, k=k, laplacian_normalization = laplacian_normalization)
    labels = soft_hfs(X, Y_masked, c_l , c_u, laplacian_regularization, var, eps, k, laplacian_normalization)
    
    
    """
    Visualize results
    """
    plot_classification(X, Y, labels,  var=var, eps=0, k=k)
    accuracy = np.mean(labels == np.squeeze(Y))
    
    return accuracy

    




def soft_hfs(X, Y, c_l, c_u, laplacian_regularization, var=1, eps=0, k=0, laplacian_normalization=""):
    """
    TO BE COMPLETED.

    Function to perform soft (unconstrained) HFS


    :param X: (n x m) matrix of m-dimensional samples
    :param Y: (n x 1) vector with nodes labels [1, ... , num_classes] (0 is unlabeled)
    :param c_l: coefficients for C matrix
    :param c_u: coefficients for C matrix
    :param laplacian_regularization:
    :param var:
    :param eps:
    :param k:
    :param laplacian_normalization:
    :return: labels, class assignments for each of the n nodes
    """
    num_samples = np.size(Y, 0)
    Cl = np.unique(Y)
    num_classes = len(Cl)-1

    """
    Compute the target y for the linear system  
    y = (n x num_classes) target vector 
    l_idx = (l x num_classes) vector with indices of labeled nodes    
    u_idx = (u x num_classes) vector with indices of unlabeled nodes 
    """
    l_idx = []
    u_idx = []
    for i in range(len(Y)):
        if Y[i] == 0:
            u_idx.append(i)
        else:
            l_idx.append(i)
    
    l_idx = np.asarray(l_idx)     
    u_idx = np.asarray(u_idx)
    
    Q = build_laplacian_regularized(X, laplacian_regularization, var=var, eps=eps, k=k, laplacian_normalization=laplacian_normalization)

    """
    compute the hfs solution, remember that you can use build_laplacian_regularized and build_similarity_graph
    f = (n x num_classes) hfs solution 
    C = (n x n) diagonal matrix with c_l for labeled samples and c_u otherwise    
    """
    
    y = np.zeros((len(Y),num_classes))
    for i in range(len(Y)):
        for j in range(1,num_classes+1):
            if Y[i] == j:
                y[i,j-1] = 1
            
                
    c=np.zeros(num_samples)
    if len(u_idx) > 0:
        c[u_idx]=np.asarray([c_u]*len(u_idx))
        c[l_idx]=np.asarray([c_l]*len(l_idx))
    else:
        c = [c_l]*len(l_idx)
    C=np.diag(c)

    """
    compute the labels assignment from the hfs solution 
    labels: (n x 1) class assignments [1, ... ,num_classes]  
    """
    f = np.linalg.inv(np.linalg.inv(C).dot(Q) + np.eye(len(C))).dot(y)
    labels = np.argmax(f, axis=1)+1
    return labels





def hard_vs_soft_hfs():
    """
    TO BE COMPLETED.

    Function to compare hard and soft HFS.
    """
    # load the data    
    in_data = loadmat(os.path.join('data', 'data_2moons_hfs.mat'))
    X = in_data['X']
    Y = in_data['Y'].squeeze()
    
    # randomly sample 20 labels
    l = 20
    # mask labels
    Y_masked = mask_labels(Y, l)

    # Create some noisy labels
    Y_masked[Y_masked != 0] = label_noise(Y_masked[Y_masked != 0], 4)
    Y_masked_2 = np.copy(Y_masked)

    """
    choose parameters
    """
    var = 1.0
    eps = 0
    k = 10
    laplacian_regularization = 0.000000001
    laplacian_normalization = ''
    c_l = 0.96
    c_u = 0.04

    """
    Compute hfs solution using soft_hfs() and hard_hfs().
    Remember to use Y_masked (the vector with some labels hidden as input and NOT Y (the vector with all labels 
    revealed)
    """
    hard_labels = hard_hfs(X, Y_masked, laplacian_regularization, var, eps, k, laplacian_normalization)
    soft_labels = soft_hfs(X, Y_masked_2, c_l, c_u, laplacian_regularization, var, eps, k, laplacian_normalization)

    plot_classification_comparison(X, Y, hard_labels, soft_labels, var=var, eps=eps, k=k)
    accuracy = [np.mean(hard_labels == np.squeeze(Y)), np.mean(soft_labels == np.squeeze(Y))]
    return accuracy




if __name__ == '__main__':
    # Question 1.1, 1.2 and 1.3
    acc = two_moons_hfs()
    print("accuracy =",acc)
    # Question 1.3
    acc = hard_vs_soft_hfs()
    print("accuracy =",acc)

