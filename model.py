import numpy as np
import pandas as pd
import config as cfg

import tensorflow as tf
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model

def compressed_interaction_net(x0, xl, D, h_k):
    x0_features = x0.get_shape()[1]
    xl_features = xl.get_shape()[1]

    xl = tf.reshape(xl, [-1, D, xl_features, 1])
    x0 = tf.reshape(x0, [-1, D, 1, x0_features])

    z = tf.matmul(xl, x0)
    reshaped_z = tf.reshape(z, [-1, D, x0_features * xl_features])
    output = Conv1D(h_k, 1, 1, kernel_regularizer=tf.keras.regularizers.l2(cfg.reg_coeff))(reshaped_z)
    output = tf.reshape(output, shape=(-1, h_k, D))
    output = Activation(cfg.activation)(output)
    return output

def build_cin(x0, D=cfg.D, T=3, h_k=cfg.default_n_filters):
    cin_layers = []
    pooling_layers = []
    xl = x0
    for layer in range(T):
        xl = compressed_interaction_net(x0, xl, D, h_k)
        cin_layers.append(xl)

        pooling = Lambda(lambda x: K.sum(x, axis=-1))(xl)
        pooling_layers.append(pooling)

    output = Concatenate(axis=-1)(pooling_layers)
    return output

def build_model(dense_features, sparse_features, total_data):
    dense_inputs = []
    for f in dense_features:
        _input = Input(shape=[1], name=f)
        dense_inputs.append(_input)

    concat_dense_inputs = Concatenate(axis=1)(dense_inputs)
    first_order_dense_layer = Dense(1)(concat_dense_inputs)

    sparse_inputs = []
    for f in sparse_features:
        _input = Input(shape=[1], name=f)
        sparse_inputs.append(_input)

    sparse_1d_embed = []
    for i, _input in enumerate(sparse_inputs):
        f = sparse_features[i]
        voc_size = total_data[f].nunique()
        _embed = Flatten()(Embedding(
            voc_size+1, 1, embeddings_regularizer=tf.keras.regularizers.l2(cfg.reg_coeff))(_input))
        sparse_1d_embed.append(_embed)
    first_order_sparse_layer = Add()(sparse_1d_embed)

    linear_part = Add()([first_order_dense_layer, first_order_sparse_layer])

    D = cfg.D
    sparse_kd_embed = []
    for i, _input in enumerate(sparse_inputs):
        f = sparse_features[i]
        voc_size = total_data[f].nunique()
        _embed = Embedding(voc_size+1, D, embeddings_regularizer=tf.keras.regularizers.l2(cfg.reg_coeff))(_input)
        sparse_kd_embed.append(_embed)
    dense_kd_embed = []
    for i, _input in enumerate(dense_inputs):
        f = dense_features[i]
        _embed = tf.expand_dims(Dense(D, kernel_regularizer=tf.keras.regularizers.l2(cfg.reg_coeff))(_input), axis=1)
        dense_kd_embed.append(_embed)

    input_kd_embed = dense_kd_embed + sparse_kd_embed
    input_feature_map = Concatenate(axis=1)(input_kd_embed)

    cin_layer = build_cin(input_feature_map)

    embed_inputs = Flatten()(input_feature_map)
    fc_layer = Dense(cfg.default_dense_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(cfg.reg_coeff))(embed_inputs)
    fc_layer = Dense(cfg.default_dense_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(cfg.reg_coeff))(fc_layer)
    fc_layer_output = Dense(cfg.default_dense_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(cfg.reg_coeff))(fc_layer)
    print(linear_part.shape, cin_layer.shape, fc_layer_output.shape)

    concat_layer = Concatenate()([linear_part, cin_layer, fc_layer_output])
    output_layer = Dense(1, activation='sigmoid')(concat_layer)

    model = Model(dense_inputs+sparse_inputs, output_layer)

    model.compile(optimizer=cfg.optimizer, loss='binary_crossentropy', metrics=['binary_crossentropy', tf.keras.metrics.AUC(name='auc')])

    return model


