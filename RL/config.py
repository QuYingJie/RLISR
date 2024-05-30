import numpy as np

class Config:
    def __init__(self):
        # doc2vec
        self.pre_doc_embedding_dict = np.loadtxt('../data/entity_doc_embedding_matrix.txt')
        # kgat
        self.pre_kgat_embedding_dict = np.loadtxt('../data/entity_graph_embedding_matrix.txt')
        # concat
        self.pre_embedding_dict = np.loadtxt('../data/entity_concat_embedding_matrix.txt')

