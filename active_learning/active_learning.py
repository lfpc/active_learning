import numpy as np
import torch
import NN_utils
from uncertainty import comparison as unc
from utils import dataloader_from_idx,uncertain_unlabeled
import NN_utils.train_and_eval as TE
from tqdm.notebook import tqdm,trange



class ActiveLearning():
    def __init__(self,model,complete_train_dataset,unc_method, train_dict, 
                labeled_idx = None,unlabeled_idx = None, reset = True, validation_dataloader = None, n_labels = 1000):
        self.net = model
        self.complete_train_dataset = complete_train_dataset
        self.labeled_idx = labeled_idx
        self.unlabeled_idx = unlabeled_idx
        self.validation_dataloader = validation_dataloader
        self.labeled_idx_per_cycle = [labeled_idx]
        self.cycle = 0
        self.accuracies_val = []
        self.train_dict = train_dict
        self.unc_method = unc_method
        self.reset = reset
        self.trainer = TE.Trainer(self.net, self.train_dict['optimizer'], self.train_dict['loss_criterion'],
                                  dataloader_from_idx(self.complete_train_dataset,self.labeled_idx, self.train_dict['batch_size']),
                                  validation_data = validation_dataloader, risk_dict = self.train_dict['risk_dict'])
        self.n_labels = n_labels
        

    def update_labels(self):
        unlabeled_dataloader = dataloader_from_idx(self.complete_train_dataset,self.unlabeled_idx, train = False)
        new_idx = uncertain_unlabeled(unlabeled_dataloader,self.unc_method, self.n_labels)
        self.labeled_idx.extend(new_idx)
        self.labeled_idx_per_cycle.append(new_idx)
        unlabeled_idx_new = [x for x in self.unlabeled_idx if x not in new_idx]
        self.unlabeled_idx = unlabeled_idx_new

    def train(self,cycles):
        with tqdm(total=cycles, leave=True) as pbar:
            while self.cycle < cycles:
                pbar.set_description(f'Loss: {self.trainer.hist_train.loss_list[-1]:.4f} | Acc_train: {self.trainer.hist_train.acc_list[-1]:.2f} | Acc_val: {self.trainer.hist_val.acc_list[-1]:.2f} | AL Progress:')
                if not self.cycle == 0:
                    self.update_labels()
                train_dataloader = dataloader_from_idx(self.complete_train_dataset,self.labeled_idx, self.train_dict['batch_size'])
                self.trainer.training_data = train_dataloader
                self.trainer.fit(train_dataloader, self.train_dict['n_epochs'],live_plot = False)
                self.cycle += 1
                pbar.update(1)
                self.test()
                if self.reset:
                    self.reset_model()
    def test(self):
        #a cada final de ciclo atualizar a acurácia no test
        acc = TE.model_acc(self.net,self.validation_dataloader)
        self.accuracies_val.append(acc)
        return acc

    def reset_model(self):
        self.net.apply(NN_utils.weight_reset)
        
