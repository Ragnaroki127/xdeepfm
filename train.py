import tensorflow as tf
import config as cfg
import dataset
from model import build_model

from tensorflow.keras.callbacks import TensorBoard

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '4, 5, 6, 7'
physical_devices = tf.config.experimental.list_physical_devices('GPU')

dense_features, sparse_features, total_data = dataset.preprocessing()

tbCallBack = TensorBoard(log_dir='./logs',  
                 histogram_freq=0, 
                 write_graph=True, 
                 write_grads=True, 
                 write_images=True,
                 embeddings_freq=0, 
                 update_freq='batch',
                 embeddings_layer_names=None, 
                 embeddings_metadata=None)

total_data = total_data.sample(frac=1.0, random_state=1)
train_data = total_data.iloc[:500000]
val_data = total_data.iloc[500000:]

train_dense = [train_data[f].values for f in dense_features]
train_sparse = [train_data[f].values for f in sparse_features]

train_label = [train_data['label'].values]

val_dense = [val_data[f].values for f in dense_features]
val_sparse = [val_data[f].values for f in sparse_features]

val_label = [val_data['label'].values]

if __name__ == "__main__":
    model = build_model(dense_features, sparse_features, total_data)
    model.fit(
        train_dense + train_sparse,
        train_label, epochs=cfg.epochs, batch_size=cfg.batch_size,
        validation_data=(val_dense + val_sparse, val_label),
        callbacks=[tbCallBack]
    )
