import tensorflow as tf

data_path = './criteo_sampled_data.csv'

D = 10

default_n_filters = 200

default_dense_units = 400

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

epochs = 15

batch_size = 4096

activation = 'linear'

reg_coeff = 0.0001
