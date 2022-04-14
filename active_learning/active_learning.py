import numpy as np
import torch
import NN_utils
from uncertainty import comparison as unc
from utils import dataloader_from_idx
import NN_utils.train_and_eval as TE



class ActiveLearning():
    def __init__(self,model,complete_train_dataset,unc_method, train_dict, 
                labeled_idx = None,unlabeled_idx = None, reset = True, test_dataloader = None):
        self.model = model
        self.net = model()
        self.complete_train_dataset = complete_train_dataset
        self.labeled_idx = labeled_idx
        self.unlabeled_idx = unlabeled_idx
        self.test_dataloader = test_dataloader
        self.labeled_idx_per_cycle = [labeled_idx]
        self.cycle = 0
        self.accuracies_test = []
        self.train_dict = train_dict
        self.unc_method = unc_method
        self.reset = reset
        self.trainer = TE.Trainer(self.net, self.train_dict['optimizer'], self.train_dict['loss_criterion'],
                                    self.complete_train_dataset, self.train_dict['risk_dict'])

    def update_labels(self):
        unlabeled_dataset = dataloader_from_idx(self.complete_train_dataset,self.unlabeled_idx, train = False)
        new_idx = self.unc_method(self.net,unlabeled_dataset)
        self.labeled_idx.extend(new_idx)
        self.labeled_idx_per_cycle.append(new_idx)

    def train(self,cycles):
        while self.cycle < cycles:
            if not self.cycle == 0:
                self.update_labels()
            train_dataloader = dataloader_from_idx(self.complete_train_dataset,self.labeled_idx, self.train_dict['batch_size'])
            self.trainer.fit(train_dataloader, self.train_dict['n_epochs'])
            self.cycle+=1
            #adicionar plot de acurácia e perda aqui
            #mudar em Trainer para algo mais automatizada, progress e gráfico
    def test(self):
        #a cada final de ciclo atualizar a acurácia no test
        self.accuracies_test = []
        pass
