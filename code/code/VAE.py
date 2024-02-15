import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd

import NSL_KDDdata

x_train , y_train= NSL_KDDdata.train_data()
x_test , y_test = NSL_KDDdata.test_data()


def train_test_shape(x_train, x_test):
    return (x_train.shape, x_test.shape)

#sampling
def sampling(args, latent_dim = 8):
    z_mean, z_log_var = args
    epsilon = tf.keras.backend.random_normal(shape=(tf.shape(z_mean)[0], latent_dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def VAE(latent_dim , x_train, epochs, original_dim):
    # Encoder
    encoder_inputs = keras.Input(shape=(original_dim,))
    x = layers.Dense(128, activation="relu")(encoder_inputs)
    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)

    # Sampling layer - samples mean and log variance of the distribution
    z = layers.Lambda(sampling)([z_mean, z_log_var], latent_dim)

    # Decoder
    decoder_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(128, activation="relu")(decoder_inputs)
    outputs = layers.Dense(original_dim, activation="sigmoid")(x)

    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    decoder = keras.Model(decoder_inputs, outputs, name="decoder")

    # Define the VAE model
    outputs = decoder(encoder(encoder_inputs)[2])
    vae = keras.Model(encoder_inputs, outputs, name="vae")
    

    # Reconstruction loss
    reconstruction_loss = keras.losses.binary_crossentropy(encoder_inputs, outputs)
    reconstruction_loss *= original_dim

    # KL loss
    kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    kl_loss = tf.reduce_sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)


    vae.compile(optimizer='adam')
    vae.fit(x_train, epochs = epochs, batch_size=64)

    return encoder, decoder


num_epohs = 20
encoder, decoder = VAE(8 , x_train, num_epohs, x_train.shape[1])

compressed_train , _, _ = encoder.predict(x_train)
compressed_test , _, _ = encoder.predict(x_test)

df = pd.DataFrame(compressed_train)
df.to_csv('data/compressed_data_train.csv')

df2 = pd.DataFrame(compressed_test)
df2.to_csv('data/compressed_data_test.csv')