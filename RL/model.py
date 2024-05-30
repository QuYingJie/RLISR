import tensorflow as tf
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import Model

class MyNet(Model):
    def __init__(self,config):
        super(MyNet, self).__init__()
        # 定义模型
        # kgat preprocessing
        self.kgat_embedding = Embedding(config.pre_kgat_embedding_dict.shape[0], 512, weights=[config.pre_kgat_embedding_dict], trainable=True, name='kgat')
        # concat preprocessing
        self.concat_embedding = Embedding(config.pre_embedding_dict.shape[0], 1112, weights=[config.pre_embedding_dict], trainable=True, name='concat')
        # enhanced preprocessing
        self.attention = GRU(512, recurrent_activation='sigmoid', return_sequences=True)
        self.pool = GlobalAveragePooling1D()
        # dqn
        self.dense1 = Dense(1024, activation='relu')
        self.dense = Dense(1322, activation='sigmoid')

    def call(self, inputs):
        # 1. input
        mashup_input = inputs[0]
        positive_input = inputs[1]
        negative_input = inputs[2]
        cascading_api_input = inputs[3]
        # 2. mashup preprocessing
        mashup_embedding = self.kgat_embedding(mashup_input)
        mashup_embedding = tf.squeeze(mashup_embedding, axis=1)
        # 3. api preprocessing
            # 3.1 positive preprocessing
        if positive_input.shape != (1,0):
            positive_concat_embedding = self.concat_embedding(positive_input)
            positive_embedding = self.attention(positive_concat_embedding)
            positive_embedding = self.pool(positive_embedding)
        else:
            positive_embedding = tf.zeros((mashup_embedding.shape[0],512))
            # 3.2 negative preprocessing
        if negative_input.shape != (1,0):
            negative_concat_embedding = self.concat_embedding(negative_input)
            negative_embedding = self.attention(negative_concat_embedding)
            negative_embedding = self.pool(negative_embedding)
        else:
            negative_embedding = tf.zeros((mashup_embedding.shape[0],512))
        # 4. state preprocessing
        state_embedding = tf.subtract(tf.add(mashup_embedding, positive_embedding), negative_embedding)
        # 5. cascading_embedding
        if cascading_api_input.shape[1] != 0:
            cascading_api_embedding = self.kgat_embedding(cascading_api_input)
            cascading_api_embedding = tf.reduce_mean(cascading_api_embedding, axis=1)
            concat_embedding = tf.stack([state_embedding, cascading_api_embedding])
            state_embedding = tf.reduce_mean(concat_embedding, axis=0)
        # 6. dqn
        output = self.dense1(state_embedding)
        output = self.dense(output)
        return output
