import sys
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Dense, Activation, LeakyReLU, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
#import sklearn
#from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.models import save_model
from tensorflow.keras.models import model_from_json
#plt.switch_backend('agg')
from tensorflow.keras import regularizers
#from tensorflow.keras.utils import multi_gpu_model
#from sklearn.decomposition import PCA

#inpt = "1000G_real_genomes/805_SNP_1000G_real.hapt" #hapt format input file

inpt = "completecases.txt" #hapt format input file
latent_size = 600 #size of noise input
alph = 0.01 #alpha value for LeakyReLU
g_learn = 0.0001 #generator learning rate
d_learn = 0.0008 #discriminator learning rate
epochs = 10001
batch_size = 32
ag_size = 216 #number of artificial genomes (haplotypes) to be created
gpu_count = 1 #number of GPUs
save_that = 200 #epoch interval for saving outputs

#For saving models
def save_mod(gan, gen, disc, epo):
    discriminator.trainable = False
    save_model(gan, epo+"_gan")
    discriminator.trainable = True
    save_model(gen, epo+"_generator")
    save_model(disc, epo+"_discriminator")


#Read input
df = pd.read_csv(inpt, sep = '\t', header=None)
df = df.sample(frac=1).reset_index(drop=True)
df_noname = df.drop(df.columns[0:2], axis=1)
df_noname = df_noname.values
df_noname = df_noname - np.random.uniform(0,0.1, size=(df_noname.shape[0], df_noname.shape[1]))
df_noname = pd.DataFrame(df_noname)

K.clear_session()

# #Make generator
generator = Sequential()
generator.add(Dense(int(df_noname.shape[1]//1.2), input_shape=(latent_size,), kernel_regularizer=regularizers.l2(0.0001)))
generator.add(LeakyReLU(alpha=alph))
generator.add(Dense(int(df_noname.shape[1]//1.1), kernel_regularizer=regularizers.l2(0.0001)))
generator.add(LeakyReLU(alpha=alph))
generator.add(Dense(df_noname.shape[1], activation = 'tanh'))

# #Make discriminator
# discriminator = Sequential()
# discriminator.add(Dense(df_noname.shape[1]//2, input_shape=(df_noname.shape[1],), kernel_regularizer=regularizers.l2(0.0001)))
# discriminator.add(LeakyReLU(alpha=alph))
# discriminator.add(Dense(df_noname.shape[1]//3, kernel_regularizer=regularizers.l2(0.0001)))
# discriminator.add(LeakyReLU(alpha=alph))
# discriminator.add(Dense(1, activation = 'sigmoid'))
# #if gpu_count > 1:
# #    discriminator = multi_gpu_model(discriminator, gpus=gpu_count)
# discriminator.compile(optimizer=Adam(lr=d_learn), loss='binary_crossentropy')
# #Set discriminator to non-trainable
# discriminator.trainable = False

# #Make GAN
# gan = Sequential()
# gan.add(generator)
# gan.add(discriminator)
# #if gpu_count > 1:
# #    gan = multi_gpu_model(gan, gpus=gpu_count)
# gan.compile(optimizer=Adam(lr=g_learn), loss='binary_crossentropy')

# y_real, y_fake = np.ones([batch_size, 1]), np.zeros([batch_size, 1])
# X_real = df_noname.values

# losses = []
# batches = len(X_real)//batch_size

# #Training iteration
# for e in range(epochs):
#     for b in range(batches):
#         X_batch_real = X_real[b*batch_size:(b+1)*batch_size] #get the batch from real data
#         latent_samples = np.random.normal(loc=0, scale=1, size=(batch_size, latent_size)) #create noise to be fed to generator
#         X_batch_fake = generator.predict_on_batch(latent_samples) #create batch from generator using noise as input

#         #train discriminator, notice that noise is added to real labels
#         discriminator.trainable = True
#         d_loss = discriminator.train_on_batch(X_batch_real, y_real - np.random.uniform(0,0.1, size=(y_real.shape[0], y_real.shape[1])))
#         d_loss += discriminator.train_on_batch(X_batch_fake, y_fake)

#         #make discriminator non-trainable and train gan
#         discriminator.trainable = False
#         g_loss = gan.train_on_batch(latent_samples, y_real)

#     losses.append((d_loss, g_loss))
#     print("Epoch:\t%d/%d Discriminator loss: %6.4f Generator loss: %6.4f"%(e+1, epochs, d_loss, g_loss))
#     if e%save_that == 0 or e == epochs:

#         #Save models
#         save_mod(gan, generator, discriminator, str(e))

#         #Create AGs
#         latent_samples = np.random.normal(loc=0, scale=1, size=(ag_size, latent_size))
#         generated_genomes = generator.predict(latent_samples)
#         generated_genomes[generated_genomes < 0] = 0
#         generated_genomes = np.rint(generated_genomes)
#         generated_genomes_df = pd.DataFrame(generated_genomes)
#         generated_genomes_df = generated_genomes_df.astype(int)
#         gen_names = list()
#         for i in range(0,len(generated_genomes_df)):
#                 gen_names.append('AG'+str(i))
#         generated_genomes_df.insert(loc=0, column='Type', value="AG")
#         generated_genomes_df.insert(loc=1, column='ID', value=gen_names)
#         generated_genomes_df.columns = list(range(generated_genomes_df.shape[1]))
#         df.columns = list(range(df.shape[1]))

#         #Output AGs in hapt format
#         generated_genomes_df.to_csv(str(e)+"_output.hapt", sep=" ", header=False, index=False)

#         #Output lossess
#         pd.DataFrame(losses).to_csv(str(e)+"_losses.txt", sep=" ", header=False, index=False)
#         fig, ax = plt.subplots()
#         plt.plot(np.array([losses]).T[0], label='Discriminator')
#         plt.plot(np.array([losses]).T[1], label='Generator')
#         plt.title("Training Losses")
#         plt.legend()
#         fig.savefig(str(e)+'_loss.pdf', format='pdf')

#         #Make PCA
#         df_pca = df.drop(df.columns[1], axis=1)
#         df_pca.columns = list(range(df_pca.shape[1]))
#         df_pca.iloc[:,0] = 'Real'
#         generated_genomes_pca = generated_genomes_df.drop(generated_genomes_df.columns[1], axis=1)
#         generated_genomes_pca.columns = list(range(df_pca.shape[1]))
#         df_all_pca = pd.concat([df_pca, generated_genomes_pca])
#         pca = PCA(n_components=2)
#         PCs = pca.fit_transform(df_all_pca.drop(df_all_pca.columns[0], axis=1))
#         PCs_df = pd.DataFrame(data = PCs, columns = ['PC1', 'PC2'])
#         PCs_df['Pop'] = list(df_all_pca[0])
#         fig = plt.figure(figsize = (10,10))
#         ax = fig.add_subplot(1,1,1)
#         ax.set_xlabel('PC1')
#         ax.set_ylabel('PC2')
#         pops = ['Real', 'AG']
#         colors = ['r', 'b']
#         for pop, color in zip(pops,colors):
#             ix = PCs_df['Pop'] == pop
#             ax.scatter(PCs_df.loc[ix, 'PC1']
#                        , PCs_df.loc[ix, 'PC2']
#                        , c = color
#                        , s = 50, alpha=0.2)
#         ax.legend(pops)
#         fig.savefig(str(e)+'_pca.pdf', format='pdf')
