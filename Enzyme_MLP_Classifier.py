import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import string
from sklearn.preprocessing import OneHotEncoder
import functools
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger
from torch.optim import lr_scheduler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import Embedding
from torch import LongTensor
from torch.autograd import Variable
import numpy as np
#Define Enzyme Dataset
class Enzyme_Dataset(Dataset):
    def __init__(self, file_name, pre_pad=True, num_samples=2000, device='cuda:0'):
        #read csv file and load row data into variables
        file_out = pd.read_csv(file_name)

        if num_samples > -1:
            self.x_seq = file_out.iloc[0:num_samples,1].values
            self.x_creature = file_out.iloc[0:num_samples,2].values
            self.y_label = file_out.iloc[0:num_samples,3]
        else:
            self.x_seq = file_out.iloc[:, 1].values
            self.x_creature = file_out.iloc[:, 2].values
            self.y_label = file_out.iloc[:, 3]
        self.pre_pad = pre_pad

        #get maximum length seq for padding
        #self.maximum_length = max([len(i) for i in self.x_seq])
        self.maximum_length = 1234

        #convert characters to int
        amino_acids = ['A','B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T','U', 'V', 'W','X',
                       'Y','Z']
        char_list = dict(enumerate(list(amino_acids)))
        char2int = {ch: i  for i, ch in char_list.items()}
        self.x_seq = list(map(lambda i: torch.tensor([(int(char2int[ch])+1)/20 for ch in i]), self.x_seq))
        self.sequence_lengths = [len(seq) for seq in self.x_seq]

        #convert labels to int
        self.x_creature = list(map(lambda label: torch.tensor(int(label[label.rfind('e')+1:])), self.x_creature))
        self.y_label = list(map(lambda label: torch.tensor(int(label[label.rfind('s')+1:])), self.y_label))

    def __len__(self):
        return len(self.y_label)



    def __getitem__(self, idx):
        # helper function to pre-pad amino acid sequence to be of equal lengths
        def zero_pad_sequence(sequence, length, pre=False):
            result = torch.zeros(length)
            sequence_length = len(sequence)
            if pre:
                result[length - sequence_length:] = sequence
            else:
                result[0:sequence_length] = sequence
            return result

        # helper function to one-hot-encode creature labels and enzyme labels
        def one_hot_encoding(i, vec_size):
            #print(i)
            vector = torch.zeros(vec_size)
            vector[i] = 1.0
            return vector
        return zero_pad_sequence(self.x_seq[idx],1234, pre=True), one_hot_encoding(self.x_creature[idx], 10), self.y_label[idx]




        #return zero_pad_sequence(sequence=self.x_seq[idx], length=self.maximum_length, pre=self.pre_pad), one_hot_encoding(self.x_creature[idx], 10), self.y_label[idx]#one_hot_encoding(self.y_label[idx],20)

#test_dataset = Enzyme_Dataset(file_name='/home/lucas/PycharmProjects/Enzyme_Classification/csv_files/train_70.csv',pre_pad=True, num_samples=1000)
#test_seq, test_creat, _ = test_dataset.__getitem__(1)
#print('{} {} {}'.format(test_seq.size(), test_creat.size(), torch.cat((test_seq, test_creat),0).size()))
#print(test_dataset.__len__())

class GRU_model(pl.LightningModule):
    def __init__(self, cfg):
        super(GRU_model, self).__init__()
        self.cfg = cfg


        self.fc_1 = nn.Sequential(
            nn.Linear(in_features=1244, out_features=1244),
            nn.BatchNorm1d(1244),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(in_features=1244, out_features=622),
            #nn.BatchNorm1d(1244),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        self.fc_3 = nn.Sequential(
            nn.Linear(in_features=622, out_features=311),
            #nn.BatchNorm1d(622),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        self.fc_4 = nn.Sequential(
            nn.Linear(in_features=311, out_features=128),
            #nn.BatchNorm1d(311),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        self.fc_class = nn.Sequential(
            nn.Linear(in_features=128, out_features=20)
        )

        self.correct = []
        self.count = []

        self.correct_test = []
        self.count_test = []


    def forward(self,x):
        #print(x.size())
        out = self.fc_1(x)
        out = self.fc_2(out)
        out = self.fc_3(out)
        out = self.fc_4(out)

        out = self.fc_class(out)
        return out

    def configure_optimizers(self):
        if self.cfg.model.optimizer == 0:
            return optim.Adam(self.parameters(),
                              lr=self.cfg.model.initial_learning_rate,
                              weight_decay=1e-6,
                              amsgrad=True)

        elif self.cfg.model.optimizer == 1:
            return optim.RMSprop(self.parameters(),
                                 lr=self.cfg.model.initial_learning_rate,
                                 weight_decay=self.cfg.model.weight_decay,
                                 alpha=0.95,
                                 momentum=0.5)
        elif self.cfg.model.optimizer == 3:
            return optim.SGD(self.parameters(),
                             lr=self.cfg.model.initial_learning_rate,
                             weight_decay=self.cfg.model.weight_decay,
                             momentum=0.9)

    def train_dataloader(self):
        return DataLoader(Enzyme_Dataset(file_name=self.cfg.data.train.path,
                                           pre_pad=True,
                                           num_samples=self.cfg.data.train.num_samples),
                            batch_size=self.cfg.data.train.batch_size,
                            shuffle=True,
                            num_workers=4,
                            pin_memory=True,
                            prefetch_factor=4)


    def val_dataloader(self):
        return  DataLoader(Enzyme_Dataset(file_name=self.cfg.data.eval.path,
                                          pre_pad=True,
                                          num_samples=self.cfg.data.eval.num_samples),
                           batch_size=self.cfg.data.eval.batch_size,
                           shuffle=False,
                           num_workers=4,
                           pin_memory=True)

    def test_dataloader(self):
        return DataLoader(Enzyme_Dataset(file_name=self.cfg.data.eval.path,
                                         pre_pad=True,
                                         num_samples=self.cfg.data.eval.num_samples),
                          batch_size=self.cfg.data.eval.batch_size,
                          shuffle=False,
                          num_workers=4,
                          pin_memory=True)

    def on_train_epoch_start(self):
        self.correct = []
        self.count = []
        self.hidden_state = 0



    def training_step(self, batch, batch_nb):
        x_seq, x_creature, y_label = batch
        output = self.forward(torch.cat((x_seq, x_creature), dim=1))
        pred = output.detach().max(1, keepdim=True)[1]
        self.correct.append(pred.eq(y_label.view_as(pred)).sum().item())
        self.count.append(len(pred))
        loss = F.cross_entropy(input=output, target=y_label, reduction='sum')
        self.log('train_loss',loss)
        return {'loss':loss}

    def on_train_epoch_end(self, outputs):
        print("\n THE TRAIN ACCURACY FOR THIS EPOCH IS {} \n".format(sum(self.correct)/sum(self.count)))
        self.log('train_acc',sum(self.correct)/sum(self.count))


    def on_validation_epoch_start(self):
        self.correct_test = []
        self.count_test = []
        self.hidden_state = 0

    def on_validation_epoch_end(self):
        print("\n THE TEST ACCURACY FOR THIS EPOCH IS {} \n".format(sum(self.correct_test) / sum(self.count_test)))
        self.log('val_acc', sum(self.correct_test)/sum(self.count_test))

    def validation_step(self, batch, batch_nb):
        x_seq, x_creature, y_label = batch
        output = self.forward(torch.cat((x_seq, x_creature), dim=1))
        pred = output.detach().max(1, keepdim=True)[1]
        self.correct_test.append(pred.eq(y_label.view_as(pred)).sum().item())
        self.count_test.append(len(pred))
        loss = F.cross_entropy(input=output, target=y_label, reduction='sum')
        self.log('val_loss',loss)
        return {'val_loss': loss, 'log': {'val_loss': loss}}

    def on_test_epoch_start(self):
        self.correct_test = []
        self.count_test = []
        self.hidden_state = 0

    def test_step(self, batch, batch_nb):
        x_seq, x_creature, y_label = batch
        output = self.forward(torch.cat((x_seq, x_creature), dim=1))
        pred = output.detach().max(1, keepdim=True)[1]
        self.correct.append(pred.eq(y_label.view_as(pred)).sum().item())
        self.count.append(len(pred))
        loss = F.cross_entropy(input=output, target=y_label)
        return {'test_loss':loss, 'test_acc':self.accuracy}

    def on_test_epoch_end(self):
        print("\n THE TEST ACCURACY FOR THIS EPOCH IS {} \n".format(sum(self.correct_test) / sum(self.count_test)))

    def validation_epoch_end(self, outputs):
        val_loss_mean = sum([o['val_loss'] for o in outputs]) / len(outputs)
        # show val_acc in progress bar but only log val_loss
        results = {'progress_bar': {'val_loss': val_loss_mean.item()}, 'log': {'val_loss': val_loss_mean.item()},
                   'val_loss': val_loss_mean.item()}
        return results


@hydra.main(config_path='/home/lucas/PycharmProjects/Enzyme_Classification/LSTM_only_config.yaml')
def main(cfg: DictConfig) -> None:
    LearningScheduler = lambda epoch: epoch // 10

    wandb_logger = WandbLogger(name=cfg.model.exp_name, project='enzyme_classification')
    seed_everything(100)
    cuda = 1 if torch.cuda.is_available() else 0
    model = GRU_model(cfg)

    trainer = pl.Trainer(logger=wandb_logger if cfg.model.wandb else None,
                         default_root_dir='/home/lucas/PycharmProjects/Enzyme_Classification/models',
                         max_epochs=cfg.model.n_epochs,
                         fast_dev_run= False,
                         track_grad_norm=2,
                         gpus=1)#, early_stop_callback=None)
    trainer.fit(model)



if __name__ == '__main__':
    main()
