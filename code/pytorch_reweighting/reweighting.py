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
from collections import Counter
from utils import plot_images

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default=MNIST, 
                        choices=[MNIST, CIFAR, 'custom'])
parser.add_argument('--data_path', type=str, default=None)
parser.add_argument('--alpha', type=float, default=0.01)
parser.add_argument('--batch_size', type=int, default=100)

class Reweighting():
    def __init__(self, data_loader=None, val_loader=None, test_loader=None, build_model=None, loss=None):
        # to be defined during setup_model if a defined dataset
        self.data_loader = data_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.build_model = build_model
        self.loss = loss

        self.dataset_name = None

        self.predict = lambda output: output # predict function, to be overriden for testing if necessary
        self.label = lambda output: output # labeling function to be overridden if necessary

    def setup_data(self, dataset_name=None, classes=[9,4], weights=[0.995,0.005], 
            n_items=5000, n_test_items=2000, n_vals_per_class=[5,5],
            test_classes=None, test_weights=None, batch_size=100, train_type='reweight'):
        print("="*50)
        print(f"Getting data loader for {dataset_name}")
        self.dataset_name = dataset_name
        self.classes = classes
        self.batch_size = batch_size

        self.test_classes = test_classes or classes 
        self.test_weights = test_weights or [1.0] * len(self.classes)

        if self.dataset_name is not None:
            (self.data_loader, self.val_loader) = get_data_loader(self.dataset_name, self.batch_size, n_items=n_items, 
                classes=classes, weights=weights, n_vals_per_class=n_vals_per_class, mode="train", train_type=train_type)
            (self.test_loader, _) = get_data_loader(self.dataset_name, self.batch_size, n_items=n_test_items,
                classes=self.test_classes, weights=self.test_weights, mode="test", train_type=train_type)
            kwargs = { 'n_out': len(classes)}
            self.build_model = get_build_model(self.dataset_name, kwargs)
        else:
            assert self.data_loader is not None
            assert self.test_loader is not None
            assert self.build_model is not None

        print("Data loader initialized")
        print("="*50)
            

    def setup_model(self):
        print("="*50)
        print(f"Setting up model for dataset {self.dataset_name}")

        if self.dataset_name in [MNIST, CIFAR]:
            self.loss = F.cross_entropy
            self.label = lambda output: torch.tensor(list(map(self.classes.index, output.int())))
            def predict(output):
                vals, inds = torch.max(F.log_softmax(output, dim=1), 1)                
                return inds
            self.predict = predict

            '''
            # Binary classification case
            self.loss = F.binary_cross_entropy_with_logits
            self.label = lambda output: (output == self.classes[1]).int().float()
            self.predict = lambda output: (F.sigmoid(output) > 0.5).int()
            '''
        elif self.dataset_name is None:
            assert self.loss is not None

        print("Model setup finished")
        print("="*50)


    def train(self, num_iter=8000, learning_rate=0.001):
        '''
            perform regular training
        '''
        print("="*50)
        print("Starting regular training")
        self.net = self.build_model()
        opt = torch.optim.SGD(self.net.params(), lr=learning_rate)

        net_losses = []
        plot_step = 100
        net_l = 0

        smoothing_alpha = 0.9
        self.accuracy_log = []
        self.accuracy_log_by_class = [[np.array([0,0])] for i in range(len(self.classes))]

        self.evaluate_and_record(0, None, net_losses)

        for i in tqdm(range(1, num_iter + 1)):
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
                self.evaluate_and_record(i, None, net_losses)

        print("="*50)
        print("train finished")
        self.evaluate(by_class=True)

    def train_reweighted(self, num_iter=8000, learning_rate=0.001, alpha=0.01, show_weights=False):
        print("="*50)
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
        self.accuracy_log = []
        self.accuracy_log_by_class = [[np.array([0,0])] for i in range(len(self.classes))]
        img_weights = None

        self.evaluate_and_record(0, meta_losses, net_losses)

        for i in tqdm(range(1, num_iter + 1)):
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
            meta_net.update_params(alpha, source_params=grads)
            
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

            img_weights = (X_train, y_train, y_train_hat, w)

            if (i) % plot_step == 0:
                self.evaluate_and_record(i, meta_losses, net_losses)

        print("train_reweighted finished")
        print("="*50)
        self.evaluate(by_class=True)

        if show_weights:
            self.display_weights(img_weights)

    def evaluate(self, by_class=False):
        self.net.eval()
        counter = Counter()
        acc_by_class = None if by_class == False else [0] * len(self.classes)
        num_by_class = None if by_class == False else [0] * len(self.classes)

        acc = []
        for _, (x_test, y_test) in enumerate(self.test_loader):
            y_test_label = self.label(y_test)
            x_test = to_var(x_test, requires_grad=False)
            y_test_label = to_var(y_test_label, requires_grad=False)

            y_hat = self.net(x_test)
            y_hat = self.predict(y_hat)

            # TODO - currently only works for classification with integer classes
            acc.append((y_hat.int() == y_test_label.int()).float())

            if by_class:
                y_hat_inds = list(y_hat.int().numpy())
                y_test_inds = list(y_test_label.int().numpy())

                for y_true_ind, y_predicted_ind in zip(y_test_inds, y_hat_inds):
                    num_by_class[y_true_ind] += 1
                    acc_by_class[y_true_ind] += (y_true_ind == y_predicted_ind)
                    counter[(self.classes[y_true_ind], self.classes[y_predicted_ind])] += 1

        accuracy = torch.cat(acc, dim=0).mean()

        if by_class:
            print("(Actual, Predicted): count")
            print(counter)
            acc_by_class = np.divide(acc_by_class, num_by_class)

        return accuracy, acc_by_class

    def evaluate_and_record(self, iter_ind, meta_losses, net_losses):
        accuracy, acc_by_class = self.evaluate(by_class=True)
        self.accuracy_log.append(np.array([iter_ind, accuracy])[None])

        IPython.display.clear_output()
        fig, axes = plt.subplots(1, 2, figsize=(13,5))
        ax1, ax2 = axes.ravel()

        ax1.plot(net_losses, label='net_losses')
        if meta_losses is not None:
            ax1.plot(meta_losses, label='meta_losses')
        ax1.set_ylabel("Losses")    
        ax1.set_xlabel("Iteration")
        ax1.legend()

        acc_log = np.concatenate(self.accuracy_log, axis=0)
        ax2.plot(acc_log[:,0],acc_log[:,1], label='Average')
        if acc_by_class is not None:
            for ind, log in enumerate(self.accuracy_log_by_class):
                log.append(np.array([iter_ind, acc_by_class[ind]]))
                acc_log_by_class = np.stack(log, axis=0)
                ax2.plot(acc_log_by_class[1:,0], acc_log_by_class[1:,1], 
                    linestyle=':', label=str(self.classes[ind]))

        ax2.set_ylabel('Accuracy')
        ax2.set_xlabel('Iteration')
        ax2.legend()

        plt.show()

    def display_weights(self, img_weights):
        (X_train, y, y_hat, w) = img_weights
        y_hat = self.predict(y_hat)

        w_ordered_inds = np.argsort(-w.float().numpy())
        max_weights = w_ordered_inds[:5]
        min_weights = w_ordered_inds[-5:]
        inds = np.concatenate((max_weights, min_weights))

        imgs = np.squeeze(X_train.numpy()[inds,:,:,:].transpose([0, 2, 3, 1]),axis=3)
        cls_pred = np.array(self.classes)[y_hat.int().numpy()[inds]]
        cls_true = np.array(self.classes)[y.int().numpy()[inds]]
        weights = w.numpy()[inds]

        plot_images(imgs, cls_true, cls_pred=cls_pred, weight=weights)

    def run(self, train_type='reweight', classes=[9,4], weights=[0.995,0.005]):
        self.setup_data()

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