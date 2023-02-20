import tensorflow as tf
import numpy as np


class TFGraphInterface(object):
    def __init__(self):
        pass

    @staticmethod
    def convert_sparse_mat_to_tensor(adj):
        row, col = adj.nonzero()
        indices = np.array(list(zip(row, col)))
        # tf中SparseTensor用于创建稀疏张量；pytorch使用sparse_coo_tensor函数创建稀疏张量
        adj_tensor = tf.SparseTensor(indices=indices, values=adj.data, dense_shape=adj.shape)
        return adj_tensor

    @staticmethod
    def convert_sparse_mat_to_tensor_inputs(X):
        coo = X.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        return indices, coo.data, coo.shape