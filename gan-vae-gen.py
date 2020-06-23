import os
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.datasets as datasets
from torchvision.datasets import MNIST, CIFAR10
from torchvision.utils import save_image
import numpy as np
import pdb
import matplotlib.pyplot as plt
import argparse
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
import inception_tf
import fid
import os.path as osp

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
learning_rate = 0.0002

class GAN(nn.Module):
    def __init__(self, hidden_size, latent_size, input_channels = 3):
        super(GAN, self).__init__()
        
#         D = nn.Sequential(
#             nn.Linear(image_size, hidden_size),
#             nn.LeakyReLU(0.2),
#             nn.Linear(hidden_size, hidden_size),
#             nn.LeakyReLU(0.2),
#             nn.Linear(hidden_size, 1),
#             nn.Sigmoid())

#         # Generator 
#         G = nn.Sequential(
#             nn.Linear(latent_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, image_size),
#             nn.Tanh())


        self.D = nn.Sequential(
            nn.Conv2d(input_channels, hidden_size, 4, 2, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(True),
            nn.Conv2d(hidden_size, hidden_size * 2, 4, 2, 1),
            nn.BatchNorm2d(hidden_size * 2),
            nn.ReLU(True),
            nn.Conv2d(hidden_size * 2, hidden_size * 4, 4, 2, 1),
            nn.BatchNorm2d(hidden_size * 4),
            nn.ReLU(True),
            nn.Conv2d(hidden_size * 4, 1, 4, 1, 0),
            nn.Sigmoid()
        )

        self.G = nn.Sequential(
            nn.ConvTranspose2d(latent_size, hidden_size * 4, 4, 1, 0),
            nn.BatchNorm2d(hidden_size * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_size * 4, hidden_size * 2, 4, 2, 1),
            nn.BatchNorm2d(hidden_size * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_size * 2, hidden_size, 4, 2, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_size, input_channels, 4, 2, 1),
            nn.Tanh()
        )

        # Device setting
        self.D = self.D.to(device)
        self.G = self.G.to(device)
        self.G = nn.DataParallel(self.G)
        self.D = nn.DataParallel(self.D)

        self.best_is = 0
        self.best_fid = 0
        
        # Binary cross entropy loss and optimizer
        self.criterion = nn.BCELoss()
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=learning_rate)
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=learning_rate)
        
    def reset_grad(self):
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()
        
    def log(self, message):
        print(message)
        fh = open(path + "/train.log", 'a+')
        fh.write(message + '\n')
        fh.close()
        
    def compute_inception_score(self, samples):
        IS_mean, IS_std = inception_tf.get_inception_score(np.array(samples), splits=10,
                                                           batch_size=100, mem_fraction=1)
        self.log('Inception score: {} +/- {}'.format(IS_mean, IS_std))
        return IS_mean, IS_std

    def compute_fid(self, samples):
        fid_score = fid.compute_fid(osp.join(inception_cache_path, 'stats.npy'), samples,
                                    inception_cache_path, dataloader)
        self.log('FID score: {}'.format(fid_score))
        return fid_score
    
    def compute_inception_fid(self):
        samples = []
        num_batches = int(50000 / batch_size)
        for batch in range(num_batches):
            with torch.no_grad():
                z = torch.randn(batch_size, latent_size, 1, 1).to(device)
                gen = self.G(z)
#                 gen = gen.view(gen.size(0), 3, 28, 28)
#                 gen = self.denorm(gen)
                gen = gen * 0.5 + 0.5
                gen = gen * 255.0
                gen = gen.cpu().numpy().astype(np.uint8)
                gen = np.transpose(gen, (0, 2, 3, 1))
                samples.extend(gen)

        IS_mean, IS_std = self.compute_inception_score(samples)
        fid = self.compute_fid(samples)
        self.log('IS: {} +/- {}'.format(IS_mean, IS_std))
        self.log('FID: {}'.format(fid))
        if self.best_is < IS_mean:
            self.best_is = IS_mean
            self.best_is_std = IS_std
            self.best_fid = fid
        if self.best_fid > fid:
            self.best_fid = fid
        self.log('Best IS: {} +/- {}'.format(self.best_is, self.best_is_std))
        self.log('Best FID: {}'.format(self.best_fid))    
            
    def save(self, model_path):
        self.log(f'Saving GAN as {model_path}')
        model_state = {}
        model_state['gan'] = self.state_dict()
        model_state['g_optimizer'] = self.g_optimizer.state_dict()
        model_state['d_optimizer'] = self.d_optimizer.state_dict()
        torch.save(model_state, model_path)
        
    def load(self, model_path, prune_gen = True, prune_disc = True):
        self.log(f'Loading saved GAN named {model_path}')
        model_state = torch.load(model_path)
        new_state = {}
        for name in model_state['gan']:
            if ("G" in name and "0" not in name and prune_gen) or ("D" in name and "9" not in name and prune_disc):
                self.log(f"loading {name}")
                new_state[name] = model_state['gan'][name]
            else:
                new_state[name] = self.state_dict()[name]
        self.load_state_dict(new_state)
        if prune_gen:
            self.g_optimizer.load_state_dict(model_state['g_optimizer'])
        if prune_disc:
            self.d_optimizer.load_state_dict(model_state['d_optimizer'])
    
    def load_vae(self, vae_init_path, vae_trained_path):
        vae_init = torch.load(vae_init_path)['vae']
        vae_trained = torch.load(vae_trained_path)
        model = self.state_dict()
        zeros = 0
        total = 0
        masks = {}
        for name in vae_init:
            gan_layer = None
#             if 'encoder.' in name and '9' not in name and '10' not in name and '12' not in name and '13' not in name:
#                 gan_layer = name.replace('encoder', 'D.module')
            if 'decoder.' in name and '4' in name:
                gan_layer = name.replace('decoder', 'G.module')
                gan_layer = gan_layer.replace('4', '1')
            if 'decoder.' in name and '6' in name:
                gan_layer = name.replace('decoder', 'G.module')
                gan_layer = gan_layer.replace('6', '3')
            if 'decoder.' in name and '7' in name:
                gan_layer = name.replace('decoder', 'G.module')
                gan_layer = gan_layer.replace('7', '4')
            if 'decoder.' in name and '9' in name:
                gan_layer = name.replace('decoder', 'G.module')
                gan_layer = gan_layer.replace('9', '6')
            if 'decoder.' in name and '10' in name:
                gan_layer = name.replace('decoder', 'G.module')
                gan_layer = gan_layer.replace('10', '7')
            if 'decoder.' in name and '12' in name:
                gan_layer = name.replace('decoder', 'G.module')
                gan_layer = gan_layer.replace('12', '9')
            if gan_layer:
                model[gan_layer] = vae_init[name]
                masks[gan_layer] = vae_trained[name].abs().gt(0).int()
                pruned = masks[gan_layer].numel() - masks[gan_layer].nonzero().size(0)
                tot = masks[gan_layer].numel()
                frac = pruned / tot
                self.log(f"{gan_layer} : {pruned} / {tot}  {frac}")
                zeros += pruned
                total += tot
        self.load_state_dict(model)
        self.log(f"Fraction of weights pruned = {zeros}/{total} = {zeros/total}")
        self.masks = masks       
        
        
    def one_shot_prune(self, pruning_perc, prune_gen = True, prune_disc = True, layer_wise = False, trained_original_model_state = None):
        self.log(f"************pruning {pruning_perc} of the network****************")
        self.log(f"Layer-wise pruning = {layer_wise}")
        model_state = None
        if trained_original_model_state:
            model_state = torch.load(trained_original_model_state)
            
        if pruning_perc > 0:
            masks = {}
            zeros = 0
            total = 0
            flat_model_weights = np.array([])
            for name in model_state:
                if ("G" in name and "0" not in name and prune_gen) or ("D" in name and "9" not in name and prune_disc):
                    layer_weights = model_state[name].data.cpu().numpy()
                    flat_model_weights = np.concatenate((flat_model_weights, layer_weights.flatten()))
            global_threshold = np.percentile(abs(flat_model_weights), pruning_perc)

            self.log("VAE layer-wise pruning percentages")
            for name in model_state:
                if ("G" in name and "0" not in name and prune_gen) or ("D" in name and "9" not in name and prune_disc):
                    threshold = global_threshold
                    if layer_wise:
                        layer_weights = model_state[name].data.cpu().numpy()
                        threshold = np.percentile(abs(layer_weights), pruning_perc)
                    masks[name] = model_state[name].abs().gt(threshold).int()
                    pruned = masks[name].numel() - masks[name].nonzero().size(0)
                    tot = masks[name].numel()
                    frac = pruned / tot
                    self.log(f"{name} : {pruned} / {tot}  {frac}")
                    zeros += pruned
                    total += tot
            self.log(f"Fraction of weights pruned = {zeros}/{total} = {zeros/total}")
            self.masks = masks
    
    def get_percent(self, total, percent):
        return (percent/100)*total


    def get_weight_fractions(self, number_of_iterations, percent):
        percents = []
        for i in range(number_of_iterations+1):
            percents.append(self.get_percent(100 - sum(percents), percent))
        self.log(f"{percents}")
        weight_fractions = []
        for i in range(1, number_of_iterations+1):
            weight_fractions.append(sum(percents[:i]))

        self.log(f"Weight fractions: {weight_fractions}")

        return weight_fractions
    
    def iterative_prune(self, init_state, 
                        trained_original_model_state, 
                        number_of_iterations, 
                        percent = 20, 
                        init_with_old = True,
                        prune_gen = True,
                        prune_disc = True):
        
        
        weight_fractions = self.get_weight_fractions(number_of_iterations, percent)       
        self.log("***************Iterative Pruning started. Number of iterations: {} *****************".format(number_of_iterations))
        for pruning_iter in range(0, number_of_iterations):
            self.log("Running pruning iteration {}".format(pruning_iter))
            self.__init__(hidden_size = hidden_size, latent_size = latent_size)
            self = self.to(device)
            trained_model = trained_original_model_state
            if pruning_iter != 0:
                trained_model = path + "/"+ "end_of_" + str(pruning_iter - 1) + '.pth'
            
            self.one_shot_prune(weight_fractions[pruning_iter], 
                                trained_original_model_state = trained_model, 
                                prune_gen = prune_gen,
                                prune_disc = prune_disc)
            self.train(prune = True, 
                       init_state = init_state, 
                       init_with_old = init_with_old,
                       prune_gen = prune_gen,
                       prune_disc = prune_disc)
            
            torch.save(self.state_dict(), path + "/"+ "end_of_" + str(pruning_iter) + '.pth')
            
            fake_images = self.G(test_z)
            #fake_images = fake_images.view(fake_images.size(0), 1, 28, 28)
            #save_image(self.denorm(fake_images), path + '/image_' + str(pruning_iter) + '.png')
            save_image(fake_images * 0.5 + 0.5, path + '/image_' + str(pruning_iter) + '.png')
            
            torch.cuda.empty_cache()
            
        self.log("Finished Iterative Pruning")
        
        
    def mask(self, prune_gen = True, prune_disc = True):
        state = self.state_dict()
        for name in state:
            if (("G" in name and "0" not in name and prune_gen) or ("D" in name and "9" not in name and prune_disc)) and name in self.masks:
                state[name].data.mul_(self.masks[name])
        self.load_state_dict(state)
    
    def denorm(self, x):
            out = (x + 1) / 2
            return out.clamp(0, 1)

    def train(self, prune, init_state = None, init_with_old = True, prune_gen = True, prune_disc = True):
        self.log(f"Number of parameters in model {sum(p.numel() for p in self.parameters())}")
        if not prune:
            self.save(path + '/before_train.pth')
        
        if prune and init_with_old:
            self.load(init_state, 
                      prune_gen = prune_gen,
                      prune_disc = prune_disc)
        
        d_losses = np.zeros(num_epochs)
        g_losses = np.zeros(num_epochs)
        real_scores = np.zeros(num_epochs)
        fake_scores = np.zeros(num_epochs)
        
        total_step = len(dataloader)
        
        for epoch in range(num_epochs):
            for i, (images, _) in enumerate(dataloader):
                #images = images.cuda()
                mini_batch = images.size()[0]
                images = images.to(device)
                #images = images.view(mini_batch, -1).cuda()
                # Create the labels which are later used as input for the BCE loss
                real_labels = torch.ones(mini_batch).to(device)
                fake_labels = torch.zeros(mini_batch).to(device)

                # ================================================================== #
                #                      Train the discriminator                       #
                # ================================================================== #

                # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
                # Second term of the loss is always zero since real_labels == 1
                outputs = self.D(images).squeeze()
                d_loss_real = self.criterion(outputs, real_labels)
                real_score = outputs

                # Compute BCELoss using fake images
                # First term of the loss is always zero since fake_labels == 0
                z = torch.randn(mini_batch, latent_size, 1, 1).to(device)
                fake_images = self.G(z)
                outputs = self.D(fake_images).squeeze()
                d_loss_fake = self.criterion(outputs, fake_labels)
                fake_score = outputs

                # Backprop and optimize
                d_loss = d_loss_real + d_loss_fake
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()
                
                if prune:
                    self.mask(prune_gen = prune_gen,
                              prune_disc = prune_disc)

                # ================================================================== #
                #                        Train the generator                         #
                # ================================================================== #

                # Compute loss with fake images
                z = torch.randn(mini_batch, latent_size, 1, 1).to(device)
                fake_images = self.G(z)
                outputs = self.D(fake_images).squeeze()

                # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
                # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
                g_loss = self.criterion(outputs, real_labels)

                # Backprop and optimize
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()
                
                if prune:
                    self.mask(prune_gen = prune_gen,
                              prune_disc = prune_disc)
                    
                d_losses[epoch] = d_losses[epoch]*(i/(i+1.)) + d_loss.item()*(1./(i+1.))
                g_losses[epoch] = g_losses[epoch]*(i/(i+1.)) + g_loss.item()*(1./(i+1.))
                real_scores[epoch] = real_scores[epoch]*(i/(i+1.)) + real_score.mean().item()*(1./(i+1.))
                fake_scores[epoch] = fake_scores[epoch]*(i/(i+1.)) + fake_score.mean().item()*(1./(i+1.))

            self.log('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}' 
                          .format(epoch, num_epochs, i+1, total_step, d_losses[epoch], g_losses[epoch], 
                                  real_scores[epoch], fake_scores[epoch]))

            if epoch == num_epochs - 1:
                #Save sampled images
                self.compute_inception_fid()
                #sample = self.G(test_z)
                #sample = sample.view(sample.size(0), 3, 28, 28)
                #save_image(self.denorm(sample), path + '/image_{}.png'.format(epoch+1))
                #save_image(sample * 0.5 + 0.5, path + '/image_{}.png'.format(epoch+1))
                
                
        # Save the model checkpoints 
        torch.save(self.state_dict(), path + '/gan.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch VAE')

    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--image_size', default=32, type=int)
    parser.add_argument('--hidden_size', default=128, type=int)
    parser.add_argument('--latent_size', default=32, type=int)
    parser.add_argument('--dataset', default='../datasets/mnist', type=str)
    parser.add_argument('--inception_cache_path', default='./inception_cache/mnist', type=str)
    parser.add_argument('--log_path', default='./mnist', type=str)
    parser.add_argument('--init_state', default='./mnist/before_train.pth', type=str)
    parser.add_argument('--trained_original_model_state', default='./mnist/gan.pth', type=str)
    parser.add_argument('--init_with_old', default='True', type=str)
    parser.add_argument('--prune_gen', default='True', type=str)
    parser.add_argument('--prune_disc', default='True', type=str)

    args = parser.parse_args()

    if args.num_epochs != '':
         num_epochs = args.num_epochs

    if args.batch_size != '':
         batch_size = args.batch_size

    if args.lr != '':
         learning_rate = args.lr

    if args.image_size != '':
         image_size = args.image_size

    if args.hidden_size != '':
         hidden_size = args.hidden_size

    if args.latent_size != '':
         latent_size = args.latent_size

    if args.dataset != '':
         dataset = args.dataset

    if args.inception_cache_path != '':
         inception_cache_path = args.inception_cache_path

    if args.log_path != '':
         path = args.log_path

    if args.init_state != '':
         init_state = args.init_state

    if args.trained_original_model_state != '':
         trained_original_model_state = args.trained_original_model_state

    if args.init_with_old != '':
         init_with_old = args.init_with_old == 'True'

    if args.prune_gen != '':
         prune_gen = args.prune_gen == 'True'

    if args.prune_disc != '':
         prune_disc = args.prune_disc == 'True'

    img_transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ])

    if 'cifar' in dataset:
        dataset = CIFAR10(dataset, download=True, transform=img_transform)
    if 'mnist' in dataset:
        dataset = MNIST(dataset, download=True, transform=img_transform)
    if 'celeba' in dataset:
        dataset = datasets.ImageFolder(dataset, transform=img_transform)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers = 4)
    print(len(dataloader))

    fh = open(path + "/train.log", 'w')
    fh.write('Logging')
    fh.close()

    inception_tf.initialize_inception()

    #test_z = torch.load("test_input").to(device)
    test_z = torch.randn(batch_size, latent_size, 1, 1).to(device)
    torch.save(test_z, "test_input")

    model = GAN(hidden_size = hidden_size, latent_size = latent_size)
    #model.one_shot_prune(80, trained_original_model_state = trained_original_model_state)
    #model.train(prune = False, init_state = init_state, init_with_old = init_with_old)
    
    for i in range(20):
        model = GAN(hidden_size = hidden_size, latent_size = latent_size).to(device)
        model.load_vae(vae_init_path = '../vae-pytorch/celeba/before_train.pth', vae_trained_path = '../vae-pytorch/celeba_iter_1/end_of_'+str(i)+'.pth')
        model.train(prune = True, init_with_old = False)
        sample = model.G(test_z)
        save_image(sample * 0.5 + 0.5, path + '/image_{}.png'.format(i))

#     model.iterative_prune(init_state = init_state, 
#                         trained_original_model_state = trained_original_model_state, 
#                         number_of_iterations = 20, 
#                         percent = 20, 
#                         init_with_old = init_with_old,
#                         prune_gen = prune_gen,
#                         prune_disc = prune_disc)
