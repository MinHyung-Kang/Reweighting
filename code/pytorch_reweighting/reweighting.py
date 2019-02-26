# Note: a lot of code referenced from https://github.com/danieltan07/learning-to-reweight-examples

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import matplotlib.pyplot as plt
from tqdm import tqdm
import IPython
import gc
import matplotlib
import argparse

from data_loader import *
from model import *
from constants import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mnist', 
                        choices=['mnist', 'TODO', 'custom'])
parser.add_argument('--data_path', type=str, default=None)
parser.add_argument('--alpha', type=float, default=0.01)
parser.add_argument('--batch_size', type=int, default=100)

class Reweighting():
    def __init__(self, dataset_name=None, data_path=None, alpha=0.01, batch_size=100, 
            data_loader=None, val_loader=None, test_loader=None, build_model=None, loss=None):
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.alpha = alpha
        self.batch_size = batch_size

        # to be defined during setup_model if a defined dataset
        self.data_loader = data_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.build_model = build_model
        self.loss = loss

        self.predict = lambda output: output # predict function, to be overriden for testing if necessary
        self.label = lambda output: output # labeling function to be overridden if necessary

    def setup_model(self, classes=[9,4], weights=[0.995,0.005], train_type='reweight'):
        print("Getting data loader")
        if self.dataset_name is not None:
            (self.data_loader, self.val_loader) = get_data_loader(self.dataset_name, self.batch_size, classes=classes, weights=weights, mode="train", train_type=train_type)
            (self.test_loader, _) = get_data_loader(self.dataset_name, self.batch_size, classes=classes, weights=weights, mode="test", train_type=train_type)
            self.build_model = get_build_model(self.dataset_name)

        if self.dataset_name == MNIST:
            self.loss = F.binary_cross_entropy_with_logits
            self.label = lambda output: (output == classes[1]).int().float()
            self.predict = lambda output: (F.sigmoid(output) > 0.5).int()
        elif self.dataset_name == 'TODO':
            pass
        else:
            # TODO: custom model - should be defined
            assert self.data_loader is not None
            assert self.test_loader is not None
            assert self.build_model is not None
            assert self.loss is not None

    def train(self, num_iter=8000, learning_rate=0.001):
        '''
            perform regular training
        '''
        print("Starting regular training")
        self.net = self.build_model()
        opt = torch.optim.SGD(self.net.params(), lr=learning_rate)

        net_losses = []
        plot_step = 100
        net_l = 0

        smoothing_alpha = 0.9
        accuracy_log = []
        for i in tqdm(range(num_iter)):
            self.net.train()
            X_train, y_train = next(iter(self.data_loader))
            y_train = self.label(y_train)

            X_train = to_var(X_train, requires_grad=False)
            y_train = to_var(y_train, requires_grad=False)

            y_train_hat = self.net(X_train)
            cost = self.loss(y_train_hat, y_train)
            
            opt.zero_grad()
            cost.backward()
            opt.step()
            
            net_l = smoothing_alpha *net_l + (1 - smoothing_alpha)* cost.item()
            net_losses.append(net_l/(1 - smoothing_alpha**(i+1)))

            if i % plot_step == 0:
                self.evaluate_and_record(i, None, net_losses, accuracy_log)

    def train_reweighted(self, num_iter=8000, learning_rate=0.001):
        print("Starting train_reweighted")
        self.net = self.build_model()
        opt = torch.optim.SGD(self.net.params(), lr=learning_rate)

        data_val, labels_val = next(iter(self.val_loader))
        labels_val = self.label(labels_val)
        X_val = to_var(data_val, requires_grad=False)
        y_val = to_var(labels_val, requires_grad=False)
        
        meta_losses = []
        net_losses = []
        plot_step = 100

        smoothing_alpha = 0.9
        
        meta_l = 0
        net_l = 0
        accuracy_log = []

        for i in tqdm(range(num_iter)):
            self.net.train()

            # Initialize a dummy network for the meta learning of the weights
            # This helps with autograd computation
            meta_net = self.build_model()
            meta_net.load_state_dict(self.net.state_dict())
            if torch.cuda.is_available():
                meta_net.cuda()

            # Line 2 get batch of data
            X_train, y_train = next(iter(self.data_loader))
            y_train = self.label(y_train)

            X_train = to_var(X_train, requires_grad=False)
            y_train = to_var(y_train, requires_grad=False)

            # Lines 4 - 5 initial forward pass to compute the initial weighted loss
            y_train_meta_hat = meta_net(X_train)
            cost = self.loss(y_train_meta_hat, y_train, reduction='none')
            eps = to_var(torch.zeros(cost.size()))
            l_f = torch.sum(cost * eps)

            meta_net.zero_grad()
            
            # Line 6 perform a parameter update
            grads = torch.autograd.grad(l_f, (meta_net.params()), create_graph=True)
            meta_net.update_params(self.alpha, source_params=grads)
            
            # Line 8 - 10 2nd forward pass and getting the gradients with respect to epsilon
            y_val_hat = meta_net(X_val)

            l_g = self.loss(y_val_hat, y_val, reduction='mean')

            grad_eps = torch.autograd.grad(l_g, eps, only_inputs=True)[0]
            
            # Line 11 computing and normalizing the weights
            w_tilde = torch.clamp(-grad_eps, min=0)
            norm_c = torch.sum(w_tilde)

            w = w_tilde / norm_c if norm_c != 0 else w_tilde

            # Lines 12 - 14 computing for the loss with the computed weights
            # and then perform a parameter update
            y_train_hat = self.net(X_train)
            cost = self.loss(y_train_hat, y_train, reduction='none')
            l_f_hat = torch.sum(cost * w)

            opt.zero_grad()
            l_f_hat.backward()
            opt.step()

            meta_l = smoothing_alpha * meta_l + (1 - smoothing_alpha)* l_g.item()
            meta_losses.append(meta_l/(1 - smoothing_alpha**(i+1)))

            net_l = smoothing_alpha * net_l + (1 - smoothing_alpha)* l_f_hat.item()
            net_losses.append(net_l/(1 - smoothing_alpha**(i+1)))

            if i % plot_step == 0:
                self.evaluate_and_record(i, meta_losses, net_losses, accuracy_log)
                
            # return accuracy
        return np.mean(acc_log[-6:-1, 1])

    def evaluate(self):
        self.net.eval()

        acc = []
        for _, (x_test, y_test) in enumerate(self.test_loader):
            y_test = self.label(y_test)
            x_test = to_var(x_test, requires_grad=False)
            y_test = to_var(y_test, requires_grad=False)

            y_hat = self.net(x_test)
            y_hat = self.predict(y_hat)

            # TODO - currently only works for classification with integer classes
            acc.append((y_hat.int() == y_test.int()).float())

        accuracy = torch.cat(acc,dim=0).mean()

        return accuracy

    def evaluate_and_record(self, iter_ind, meta_losses, net_losses, accuracy_log):
        accuracy = self.evaluate()
        accuracy_log.append(np.array([iter_ind, accuracy])[None])

        IPython.display.clear_output()
        fig, axes = plt.subplots(1, 2, figsize=(13,5))
        ax1, ax2 = axes.ravel()

        ax1.plot(net_losses, label='net_losses')
        if meta_losses is not None:
            ax1.plot(meta_losses, label='meta_losses')
        ax1.set_ylabel("Losses")    
        ax1.set_xlabel("Iteration")
        ax1.legend()

        acc_log = np.concatenate(accuracy_log, axis=0)
        ax2.plot(acc_log[:,0],acc_log[:,1])
        ax2.set_ylabel('Accuracy')
        ax2.set_xlabel('Iteration')
        plt.show()

    def run(self, train_type='reweight', classes=[9,4], weights=[0.995,0.005]):
        self.setup_model()

        if train_type == 'reweight':
            self.train()
        else:
            self.train_reweighted()

        self.evaluate()

if __name__ == '__main__':
    args = parser.parse_args()
    model = Reweighting(args.dataset, args.data_path, args.alpha, args.batch_size)
    model.run()