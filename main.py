from __future__ import print_function
import matplotlib; matplotlib.use('Agg')
#%matplotlib inline
import torch
import numpy
import argparse
import random
numpy.random.seed(8)
torch.manual_seed(8)
torch.cuda.manual_seed(8)
from network import VaeGan
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.optim import RMSprop,Adam,SGD
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
#import progressbar
from torchvision.utils import make_grid
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from torchvision.utils import save_image

import os
from options import args
from data_loader import get_loader

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

directory = './result'
if not os.path.exists(directory):
    os.makedirs(directory)

if __name__ == "__main__":

    z_size = args.z_size
    recon_level = args.recon_level
    decay_mse = args.decay_mse
    decay_margin = args.decay_margin
    n_epochs = args.n_epochs
    lambda_mse = args.lambda_mse
    lr = args.lr
    decay_lr = args.decay_lr
    decay_equilibrium = args.decay_equilibrium

    #------------ dataloaders -------------#

    train_loader = get_loader(args.batch_size, args.train_img_dir, args.train_attr_path, args.selected_attrs, args.celeba_crop_size,
                        args.input_size[1], dataset='CelebA', mode='train', num_workers=args.num_workers)

    test_loader = get_loader(args.batch_size, args.test_img_dir, args.test_attr_path, args.selected_attrs, args.celeba_crop_size, args.input_size[1], dataset='CelebA', mode='test', num_workers=args.num_workers)

    print(len(train_loader.dataset))
    print(len(test_loader.dataset))

    net = VaeGan(z_size=z_size,recon_level=recon_level).cuda()

    #------------ margin and equilibirum -------------#

    margin = 0.35
    equilibrium = 0.68
    #mse_lambda = 1.0

    #------------ optimizers -------------#

    # an optimizer for each of the sub-networks, so we can selectively backprop
    #optimizer_encoder = Adam(params=net.encoder.parameters(),lr = lr,betas=(0.9,0.999))
    #lr_encoder = MultiStepLR(optimizer_encoder,milestones=[2],gamma=1)
    optimizer_encoder = RMSprop(params=net.encoder.parameters(),lr=lr,alpha=0.9,eps=1e-8,weight_decay=0,momentum=0,centered=False)
    lr_encoder = ExponentialLR(optimizer_encoder, gamma=decay_lr)
    
    #optimizer_decoder = Adam(params=net.decoder.parameters(),lr = lr,betas=(0.9,0.999))
    #lr_decoder = MultiStepLR(optimizer_decoder,milestones=[2],gamma=1)
    optimizer_decoder = RMSprop(params=net.decoder.parameters(),lr=lr,alpha=0.9,eps=1e-8,weight_decay=0,momentum=0,centered=False)
    lr_decoder = ExponentialLR(optimizer_decoder, gamma=decay_lr)
    
    #optimizer_discriminator = Adam(params=net.discriminator.parameters(),lr = lr,betas=(0.9,0.999))
    #lr_discriminator = MultiStepLR(optimizer_discriminator,milestones=[2],gamma=1)
    optimizer_discriminator = RMSprop(params=net.discriminator.parameters(),lr=lr,alpha=0.9,eps=1e-8,weight_decay=0,momentum=0,centered=False)
    lr_discriminator = ExponentialLR(optimizer_discriminator, gamma=decay_lr)

    #------------ training loop -------------#

    for i in range(n_epochs+1):
        print('Epoch:%s' % (i))
        for j, (x, label) in enumerate(train_loader):
            net.train()
            batch_size = len(x)

            x = Variable(x, requires_grad=False).float().cuda()

            x_tilde, disc_class, disc_layer, mus, log_variances = net(x)

            # split so we can get the different parts
            disc_layer_original = disc_layer[:batch_size]
            disc_layer_predicted = disc_layer[batch_size:-batch_size]
            disc_layer_sampled = disc_layer[-batch_size:]

            disc_class_original = disc_class[:batch_size]
            disc_class_predicted = disc_class[batch_size:-batch_size]
            disc_class_sampled = disc_class[-batch_size:]
            
            nle, kl, mse, bce_dis_original, bce_dis_predicted, bce_dis_sampled = VaeGan.loss(x, x_tilde, \
                                                                        disc_layer_original, disc_layer_predicted, disc_layer_sampled, 
                                                                        disc_class_original, disc_class_predicted, disc_class_sampled,
                                                                        mus, log_variances)

            # THIS IS THE MOST IMPORTANT PART OF THE CODE
            loss_encoder = torch.sum(kl)+torch.sum(mse)
            loss_discriminator = torch.sum(bce_dis_original) + torch.sum(bce_dis_predicted) + torch.sum(bce_dis_sampled)
            loss_decoder = torch.sum(lambda_mse * mse) - (1.0 - lambda_mse) * loss_discriminator

            # selectively disable the decoder of the discriminator if they are unbalanced
            train_dis = True
            train_dec = True
            
            if torch.mean(bce_dis_original).item() < equilibrium-margin or torch.mean(bce_dis_predicted).item() < equilibrium-margin:
                train_dis = False
            if torch.mean(bce_dis_original).item() > equilibrium+margin or torch.mean(bce_dis_predicted).item() > equilibrium+margin:
                train_dec = False
            if train_dec is False and train_dis is False:
                train_dis = True
                train_dec = True


            net.zero_grad()

            # encoder
            loss_encoder.backward(retain_graph=True)  #someone likes to clamp the grad here: [p.grad.data.clamp_(-1,1) for p in net.encoder.parameters()]
            optimizer_encoder.step()
            net.zero_grad()  # cleanothers, so they are not afflicted by encoder loss

            #decoder
            if train_dec:
                loss_decoder.backward(retain_graph=True)  #[p.grad.data.clamp_(-1,1) for p in net.decoder.parameters()]
                optimizer_decoder.step()
                net.discriminator.zero_grad()  #clean the discriminator

            #discriminator
            if train_dis:
                loss_discriminator.backward()  #[p.grad.data.clamp_(-1,1) for p in net.discriminator.parameters()]
                optimizer_discriminator.step()

            print('[%02d] encoder loss: %.5f | decoder loss: %.5f | discriminator loss: %.5f' % (i, loss_encoder, loss_decoder, loss_discriminator))

        lr_encoder.step()
        lr_decoder.step()
        lr_discriminator.step()

        margin *=decay_margin
        equilibrium *=decay_equilibrium
        if margin > equilibrium:
            equilibrium = margin
        lambda_mse *=decay_mse
        if lambda_mse > 1:
            lambda_mse=1

        for j, (x, label) in enumerate(test_loader):
            net.eval()

            x = Variable(x, requires_grad=False).float().cuda()

            out = x.data.cpu()
            out = (out + 1) / 2
            save_image(vutils.make_grid(out[:64], padding=5, normalize=True).cpu(), './result/original%s.png' % (i), nrow=8)

            out = net(x)  #out=x_tilde
            out = out.data.cpu()
            out = (out + 1) / 2
            save_image(vutils.make_grid(out[:64], padding=5, normalize=True).cpu(), './result/reconstructed%s.png' % (i), nrow=8)

            out = net(None, 100)  ##out=x_p
            out = out.data.cpu()
            out = (out + 1) / 2
            save_image(vutils.make_grid(out[:64], padding=5, normalize=True).cpu(), './result/generated%s.png' % (i), nrow=8)

            break

    exit(0)
    
    
    #------------ train LSTM -------------#

    #An example from torch, modify for our case!!
    
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()

with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    print(tag_scores)

for epoch in range(300):  # change to range that we need 
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

# See what the scores are after training
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)

    # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
    # for word i. The predicted tag is the maximum scoring tag.
    # Here, we can see the predicted sequence below is 0 1 2 0 1
    # since 0 is index of the maximum value of row 1,
    # 1 is the index of maximum value of row 2, etc.
    # Which is DET NOUN VERB DET NOUN, the correct sequence!
    print(tag_scores)
