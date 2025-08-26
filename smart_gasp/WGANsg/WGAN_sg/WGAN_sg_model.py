import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv1D, Conv2D, Dense, Flatten, Reshape,
    LeakyReLU, Dropout, UpSampling2D, UpSampling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import layers
import pickle
import numpy as np

def build_generator(png_dim1,png_dim2,input_dim):
    model = Sequential()
    model.add(Dense(png_dim1*png_dim2*input_dim, input_dim=input_dim))
    model.add(LeakyReLU(0.2))
    model.add(Reshape((png_dim1,png_dim2,input_dim)))

    model.add(Conv2D(32,2,1, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(64,2,1, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(128,2,1, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(LeakyReLU(0.3))

    model.add(Conv2D(256,2,1, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(LeakyReLU(0.5))


    model.add(Conv2D(1, 4, padding='same', activation='sigmoid'))

    return model

def build_discriminator(png_dim1,png_dim2):
    model = Sequential()
    model.add(Conv2D(32, 4, input_shape = (png_dim1,png_dim2,1)))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    model.add(Conv2D(64, 4))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    model.add(Conv2D(128, 3))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    model.add(Conv2D(256, 3))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))
    model.add(Dense(1))

    return model

gpus = tf.config.list_physical_devices('GPU')
cpus = tf.config.list_physical_devices('CPU')
if len(gpus)>0:
    device = gpus[0].name.replace('physical_device','device')
else:
    device = cpus[0].name.replace('physical_device','device')
with tf.device(device):
    class GANS(Model):
        def __init__(self,generator,discriminator,input_dim,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.input_dim = input_dim
            self.batch_size = None
            self.generator = generator
            self.discriminator = discriminator

        def compile(self,g_opt,d_opt,g_loss,d_loss,*args,**kwargs):
            super().compile(*args,**kwargs)
            self.g_opt = g_opt
            self.d_opt = d_opt
            self.g_loss = g_loss
            self.d_loss = d_loss
        def batch_data(self,data,batch_size):
            data = np.array(data)
            data_size = (len(data)//batch_size)*batch_size
            batched_data = data.reshape((data.shape[0],data.shape[1],data.shape[2],1))[0:data_size]
            self.batch_size = batch_size
            return batched_data
        def wasserstein_distance_loss(self,real_output, fake_output):
            return (tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)) #+ tf.constant(1.0, dtype=tf.float32)
        def generator_loss(self,fake_output):
            return -tf.reduce_mean(fake_output)
        def gradient_penalty(self,batch,real_output,fake_output):
            max_pixel_value = 1
            alpha = tf.random.normal([batch,1,1,1],0.0,1.0)
            diff = fake_output - (real_output/max_pixel_value)
            interpolated = (real_output/max_pixel_value) + alpha * diff
            with tf.GradientTape() as gp_tape:
                gp_tape.watch(interpolated)
                pred = self.discriminator(interpolated,training = True)
            grads = gp_tape.gradient(pred,[interpolated])[0]
            norm = tf.sqrt(tf.reduce_sum(tf.square(grads),axis = [1,2,3]))
            gp = tf.reduce_mean((norm - 1.0)**2)
            return grads,norm,gp,pred,diff,interpolated

        def train_step(self,batch):
            batch_size = self.batch_size
            input_dim = self.input_dim
            real_images = batch
            with tf.GradientTape() as d_tape:
                fake_images = self.generator(tf.random.normal((batch_size,input_dim,1)),training = True)
                yhat_real = self.discriminator((real_images), training=True)
                yhat_fake = self.discriminator((fake_images), training=True)
                yhat_realfake = tf.concat([yhat_real, yhat_fake], axis=0)
                y_realfake = tf.concat([tf.zeros_like(yhat_real), tf.ones_like(yhat_fake)], axis=0)
                noise_real = 0.3*tf.random.normal(tf.shape(yhat_real))
                noise_fake = -0.3*tf.random.normal(tf.shape(yhat_fake))
                yhat_real += noise_real
                yhat_fake += noise_fake
                y_realfake += tf.concat([noise_real, noise_fake], axis=0)
                d_loss = self.wasserstein_distance_loss(yhat_real, yhat_fake)
                gp_grads,gp_norm,gp,pred,diff,interpolated = self.gradient_penalty(batch_size,real_images,fake_images)
                total_d_loss = d_loss + gp
                combined_d_loss = total_d_loss
                dgrad = d_tape.gradient(combined_d_loss, self.discriminator.trainable_variables)
                self.d_opt.apply_gradients(zip(dgrad, self.discriminator.trainable_variables))
            with tf.GradientTape() as g_tape:
                gen_images = self.generator(tf.random.normal((batch_size,input_dim,1)), training=True)
                predicted_labels = self.discriminator(gen_images, training=True)
                total_g_loss = self.generator_loss(predicted_labels)
                combined_g_loss = total_g_loss
            ggrad = g_tape.gradient(combined_g_loss, self.generator.trainable_variables)
            self.g_opt.apply_gradients(zip(ggrad, self.generator.trainable_variables))

            return {"d_loss":total_d_loss, "g_loss":total_g_loss}

