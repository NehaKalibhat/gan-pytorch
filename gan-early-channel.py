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
from compute_flops import print_model_param_nums, print_model_param_flops

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
learning_rate = 0.0002

class EarlyBird():
    def __init__(self, percent, epoch_keep=5):
        self.percent = percent
        self.epoch_keep = epoch_keep
        self.masks = []
        self.dists = [1 for i in range(1, self.epoch_keep)]

    def pruning(self, model, percent):
        total = 0
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                total += m.weight.data.shape[0]

        bn = torch.zeros(total)
        index = 0
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                size = m.weight.data.shape[0]
                bn[index:(index+size)] = m.weight.data.abs().clone()
                index += size

        y, i = torch.sort(bn)
        thre_index = int(total * percent)
        thre = y[thre_index]
        # print('Pruning threshold: {}'.format(thre))

        mask = torch.zeros(total)
        index = 0
        for k, m in enumerate(model.modules()):
            if isinstance(m, nn.BatchNorm2d):
                size = m.weight.data.numel()
                weight_copy = m.weight.data.abs().clone()
                _mask = weight_copy.gt(thre.cuda()).float().cuda()
                mask[index:(index+size)] = _mask.view(-1)
                # print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.format(k, _mask.shape[0], int(torch.sum(_mask))))
                index += size

        # print('Pre-processing Successful!')
        return mask

    def put(self, mask):
        if len(self.masks) < self.epoch_keep:
            self.masks.append(mask)
        else:
            self.masks.pop(0)
            self.masks.append(mask)

    def cal_dist(self):
        if len(self.masks) == self.epoch_keep:
            for i in range(len(self.masks)-1):
                mask_i = self.masks[-1]
                mask_j = self.masks[i]
                self.dists[i] = 1 - float(torch.sum(mask_i==mask_j)) / mask_j.size(0)
            return True
        else:
            return False

    def early_bird_emerge(self, model):
        mask = self.pruning(model, self.percent)
        self.put(mask)
        flag = self.cal_dist()
        if flag == True:
            print(self.dists)
            for i in range(len(self.dists)):
                if self.dists[i] > 1:
                    return False
            return True
        else:
            return False
        
        
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
    
    def iterative_prune(self, 
                        number_of_iterations, 
                        percent = 20):
        weight_fractions = self.get_weight_fractions(number_of_iterations, percent)       
        self.log("***************Iterative Pruning started. Number of iterations: {} *****************".format(number_of_iterations))
        for pruning_iter in range(0, number_of_iterations):
            self.log("Running pruning iteration {}".format(pruning_iter))
            self.__init__(hidden_size = hidden_size, latent_size = latent_size)
            self = self.to(device)
            
            self.train(pruning_perc = weight_fractions[pruning_iter]/100)
            
            torch.save(self.state_dict(), path + "/"+ "end_of_" + str(pruning_iter) + '.pth')
            
            fake_images = self.G(test_z)
            save_image(fake_images * 0.5 + 0.5, path + '/image_' + str(pruning_iter) + '.png')
            
            torch.cuda.empty_cache()
            
        self.log("Finished Iterative Pruning")
        
    def train(self, pruning_perc):
        self.log(f"Number of parameters in model {sum(p.numel() for p in self.parameters())}")
        
        self.log(f'original model param: {print_model_param_nums(self)}')
        self.log(f'original model flops: {print_model_param_flops(self, image_size, True)}')
        
        
        eb = EarlyBird(pruning_perc, epoch_keep = 2)
        found_eb = False
        
        d_losses = np.zeros(num_epochs)
        g_losses = np.zeros(num_epochs)
        real_scores = np.zeros(num_epochs)
        fake_scores = np.zeros(num_epochs)
        
        total_step = len(dataloader)
        
        for epoch in range(num_epochs):
            if not found_eb and eb.early_bird_emerge(self):
                found_eb = True
                print("FOUND EARLY BIRD TICKET, Pruning model to ", pruning_perc)
                self = one_shot_prune(self, pruning_perc)

                self.log(f'pruned model param: {print_model_param_nums(self)}')
                self.log(f'pruned model flops: {print_model_param_flops(self, image_size, True)}')

            for i, (images, _) in enumerate(dataloader):
                    
                mini_batch = images.size()[0]
                images = images.to(device)
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
                sample = self.G(test_z)
                save_image(sample * 0.5 + 0.5, path + '/image_{}.png'.format(pruning_perc))
                
                
        # Save the model checkpoints 
        torch.save(self.state_dict(), path + '/gan.pth')

        
def one_shot_prune(model, pruning_perc):
    
    for name, param in model.named_parameters():
        print(name, param.data.numel())
    
    
    model.log(f"************pruning {pruning_perc} of the network****************")

    total = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]

    bn = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[index:(index+size)] = m.weight.data.abs().clone()
            index += size

    p_flops = 0
    y, i = torch.sort(bn)
    # comparsion and permutation (sort process)
    p_flops += total * np.log2(total) * 3
    thre_index = int(total * pruning_perc)
    thre = y[thre_index]

    pruned = 0
    cfg = []
    cfg_mask = []
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(thre.cuda()).float().cuda()
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            if int(torch.sum(mask)) > 0:
                cfg.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                format(k, mask.shape[0], int(torch.sum(mask))))
        elif isinstance(m, nn.MaxPool2d):
            cfg.append('M')
    print(cfg)
    # apply mask flops
    # p_flops += total

    # compare two mask distance
    p_flops += 2 * total #(minus and sum)

    pruned_ratio = pruned/total
    print("Pruned ratio: ", pruned_ratio)
    print('  + Memory Request: %.2fKB' % float(total * 32 / 1024 / 8))
    print('  + Flops for pruning: %.2fM' % (p_flops / 1e6))

    newmodel = GAN(hidden_size = hidden_size, latent_size = latent_size)

    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    for [m0, m1] in zip(model.modules(), newmodel.modules()):
        print("Layer id ", layer_id_in_cfg)
        if isinstance(m0, nn.BatchNorm2d):
            print(end_mask)
            if torch.sum(end_mask) == 0:
                continue
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):
                end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m0, nn.Conv2d):
            if torch.sum(end_mask) == 0:
                continue
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            # random set for test
            # new_end_mask = np.asarray(end_mask.cpu().numpy())
            # new_end_mask = np.append(new_end_mask[int(len(new_end_mask)/2):], new_end_mask[:int(len(new_end_mask)/2)])
            # idx1 = np.squeeze(np.argwhere(new_end_mask))

            print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w1 = None
            if m1.weight.shape[0] == 1:
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                m1.bias.data = m0.bias.data.clone()
                layer_id_in_cfg -= 1
            else:
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                w1 = w1[idx1.tolist(), :, :, :].clone()
                m1.bias.data = m0.bias.data[idx1.tolist()].clone()
                
            m1.weight.data = w1.clone()
            
        elif isinstance(m0, nn.ConvTranspose2d):
            if torch.sum(end_mask) == 0:
                continue
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            # random set for test
            # new_end_mask = np.asarray(end_mask.cpu().numpy())
            # new_end_mask = np.append(new_end_mask[int(len(new_end_mask)/2):], new_end_mask[:int(len(new_end_mask)/2)])
            # idx1 = np.squeeze(np.argwhere(new_end_mask))

            print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w1 = None
            if m1.weight.shape[0] == latent_size:
                w1 = m0.weight.data[:, idx1.tolist(), :, :].clone()
                m1.bias.data = m0.bias.data[idx1.tolist()].clone()
                layer_id_in_cfg += 1
            elif m1.weight.shape[1] == 3:
                w1 = m0.weight.data[idx0.tolist(), :, :, :].clone()
                m1.bias.data = m0.bias.data.clone()
            else:   
                w1 = m0.weight.data[:, idx1.tolist(), :, :].clone()
                m1.bias.data = m0.bias.data[idx1.tolist()].clone()
                w1 = w1[idx0.tolist(), :, :, :].clone()   
            
            m1.weight.data = w1.clone()

    #torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(args.save, 'pruned.pth.tar'))

    model = newmodel
    
    for name, param in model.named_parameters():
        print(name, param.data.numel())
        
    return model
    
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
    
#     for i in range(20):
#         model = GAN(hidden_size = hidden_size, latent_size = latent_size).to(device)
#         model.load_vae(vae_init_path = '../vae-pytorch/celeba/before_train.pth', vae_trained_path = '../vae-pytorch/celeba_iter_1/end_of_'+str(i)+'.pth')
#         model.train(prune = True, init_with_old = False)
#         sample = model.G(test_z)
#         save_image(sample * 0.5 + 0.5, path + '/image_{}.png'.format(i))

#     model.iterative_prune(number_of_iterations = 20, 
#                         percent = 20)

    model.train(0.2)