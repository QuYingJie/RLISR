import json
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import Model

MAX_DOC_LENGTH = 1
MAX_CATEGORY_LENGTH = 20
MAX_WORD_LENGTH = 3

word_embedding_mat = np.loadtxt('../data/preprocessing/word_matrix.txt')
category_embedding_mat = np.loadtxt('../data/preprocessing/category_matrix.txt')
doc_embedding_mat = np.loadtxt('../data/preprocessing/doc_matrix.txt')

class doc_process(Layer):
    def __init__(self, **kwargs):
        super(doc_process, self).__init__(**kwargs)
        self.doc_embedding = Embedding(doc_embedding_mat.shape[0], 300, weights=[doc_embedding_mat], trainable=True)

    def call(self, input):
        doc_output = self.doc_embedding(input)
        return doc_output

class category_process(Layer):
    def __init__(self, **kwargs):
        super(category_process, self).__init__(**kwargs)
        self.category_embedding = Embedding(category_embedding_mat.shape[0], 300, weights=[category_embedding_mat], trainable=True)
        self.avg = Lambda(lambda x: K.mean(x, axis=1))

    def call(self, input):
        remove = tf.convert_to_tensor([0] * 300, dtype='float32')
        category = self.category_embedding(input)
        category_mask = tf.reduce_all(tf.equal(category, remove), axis=-1)
        category_x = tf.ragged.boolean_mask(category, ~category_mask)
        category_output = self.avg(category_x)
        category_output =tf.where(tf.math.is_nan(category_output), tf.zeros_like(category_output), category_output)
        return category_output

class word_process(Layer):
    def __init__(self, **kwargs):
        super(word_process, self).__init__(**kwargs)
        self.word_embedding = Embedding(word_embedding_mat.shape[0], 300, weights=[word_embedding_mat], trainable=True)
        self.avg = Lambda(lambda x: K.mean(x, axis=1))

    def call(self, input):
        remove = tf.convert_to_tensor([0] * 300, dtype='float32')
        word = self.word_embedding(input)
        word_mask = tf.reduce_all(tf.equal(word, remove), axis=-1)
        word_x = tf.ragged.boolean_mask(word, ~word_mask)
        word_output = self.avg(word_x)
        word_output = tf.where(tf.math.is_nan(word_output), tf.zeros_like(word_output), word_output)
        return word_output

doc_input = Input(shape=(MAX_DOC_LENGTH,), dtype='int32')
doc_output = doc_process()(doc_input)

category_input = Input(shape=(MAX_CATEGORY_LENGTH,), dtype='int32')
category_output = category_process()(category_input)
category_output = Reshape((1,300))(category_output)

word_input = Input(shape=(MAX_WORD_LENGTH,), dtype='int32')
word_output = word_process()(word_input)
word_output = Reshape((1,300))(word_output)

service = tf.concat([doc_output, category_output], axis=-1)
concat = tf.concat([service, word_output], axis=-1)
flatten = Flatten()(concat)
predict_output = Dense(128, activation='tanh', name='Dense_1')(flatten)
training_output = Dense(1, activation='sigmoid', name='Dense_2')(predict_output)

model = Model([doc_input, category_input, word_input], training_output)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

with open('../data/preprocessing/positive_data.json', 'r', encoding='utf-8') as f:
    positive_data = json.load(f)
f.close()
with open('../data/preprocessing/negative_data.json', 'r', encoding='utf-8') as f:
    negative_data = json.load(f)
f.close()

random.seed(2023)
random.shuffle(positive_data)
random.seed(2023)
random.shuffle(negative_data)

train_positive_sample = positive_data[:110055]
validation_positive_sample = positive_data[110055:146740]
test_positive_sample = positive_data[146740:]
train_negative_sample = negative_data[:330020]
validation_negative_sample = negative_data[330020:440025]
test_negative_sample = negative_data[440025:]

train_sample = train_positive_sample.copy()
train_sample.extend(train_negative_sample)
validation_sample = validation_positive_sample.copy()
validation_sample.extend(validation_negative_sample)
test_sample = test_positive_sample.copy()
test_sample.extend(test_negative_sample)

random.seed(2023)
random.shuffle(train_sample)
random.seed(2023)
random.shuffle(validation_sample)
random.seed(2023)
random.shuffle(test_sample)

train_doc_sample = [row[0] for row in train_sample]
train_doc_sample = tf.convert_to_tensor(train_doc_sample, dtype='int32')
train_word_sample = [row[1] for row in train_sample]
train_word_sample = tf.convert_to_tensor(train_word_sample, dtype='int32')
train_category_sample = [row[2] for row in train_sample]
train_category_sample = tf.convert_to_tensor(train_category_sample, dtype='int32')
x_train = [train_doc_sample, train_category_sample, train_word_sample]

validation_doc_sample = [row[0] for row in validation_sample]
validation_doc_sample = tf.convert_to_tensor(validation_doc_sample, dtype='int32')
validation_word_sample = [row[1] for row in validation_sample]
validation_word_sample = tf.convert_to_tensor(validation_word_sample, dtype='int32')
validation_category_sample = [row[2] for row in validation_sample]
validation_category_sample = tf.convert_to_tensor(validation_category_sample, dtype='int32')
x_validation = [validation_doc_sample, validation_category_sample, validation_word_sample]

test_doc_sample = [row[0] for row in test_sample]
test_doc_sample = tf.convert_to_tensor(test_doc_sample, dtype='int32')
test_word_sample = [row[1] for row in test_sample]
test_word_sample = tf.convert_to_tensor(test_word_sample, dtype='int32')
test_category_sample = [row[2] for row in test_sample]
test_category_sample = tf.convert_to_tensor(test_category_sample, dtype='int32')
x_test = [test_doc_sample, test_category_sample, test_word_sample]

y_train = [row[3] for row in train_sample]
y_train = tf.convert_to_tensor(y_train, dtype='int32')
y_validation = [row[3] for row in validation_sample]
y_validation = tf.convert_to_tensor(y_validation, dtype='int32')
y_test = [row[3] for row in test_sample]
y_test = tf.convert_to_tensor(y_test, dtype='int32')

checkpoint_filepath = "../model/model"
model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath,
                                            monitor='val_accuracy',
                                            verbose=1,
                                            save_best_only=True,
                                            save_weights_only=False,
                                            mode='max')
callbacks_list = [model_checkpoint_callback]

model.fit(x=x_train, y=y_train, validation_data=(x_validation,y_validation), epochs=50, batch_size=256, callbacks=callbacks_list)

score = model.evaluate(x_test, y_test, verbose=1)
print('loss:', score[0])
print('accuracy:', score[1])

model.summary()

