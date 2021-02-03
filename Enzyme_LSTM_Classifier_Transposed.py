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
        amino_acids = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
        char_list = dict(enumerate(list(amino_acids)))
        char2int = {ch: i  for i, ch in char_list.items()}
        self.x_seq = list(map(lambda i: torch.tensor([int(char2int[ch]) for ch in i]), self.x_seq))
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
        return self.x_seq[idx], self.sequence_lengths[idx], self.x_creature[idx], self.y_label[idx]




        #return zero_pad_sequence(sequence=self.x_seq[idx], length=self.maximum_length, pre=self.pre_pad), one_hot_encoding(self.x_creature[idx], 10), self.y_label[idx]#one_hot_encoding(self.y_label[idx],20)

#test_dataset = Enzyme_Dataset(file_name='/home/lucas/PycharmProjects/Enzyme_Classification/csv_files/train_70.csv',pre_pad=True, num_samples=1000)
#test_seq, test_creat, _ = test_dataset.__getitem__(1)
#print('{} {} {}'.format(test_seq.size(), test_creat.size(), torch.cat((test_seq, test_creat),0).size()))
#print(test_dataset.__len__())

class GRU_model(pl.LightningModule):
    def __init__(self, cfg):
        super(GRU_model, self).__init__()

        self.input_dim = cfg.model.input_size
        self.hidden_dim = cfg.model.hidden_size
        self.output_dim = cfg.model.output_size
        self.n_layers = cfg.model.n_layers
        self.cfg = cfg
        self.accuracy = pl.metrics.Accuracy()
        self.hidden_state = 0

        self.embed_seq = Embedding(26, cfg.model.embed_dim_seq)
        self.embed_creature = Embedding(10, cfg.model.embed_dim_creature)

        self.gru = nn.GRU(input_size=(cfg.model.embed_dim_seq+cfg.model.embed_dim_creature),hidden_size=cfg.model.hidden_size, num_layers=self.n_layers, batch_first=True, dropout=cfg.model.dropout, bidirectional=cfg.model.bi)

        self.num_directions = (2 if cfg.model.bi else 1)
        self.fc_1 = nn.Sequential(
            nn.Linear(in_features=cfg.model.hidden_size*self.num_directions, out_features=20)
            #nn.BatchNorm1d(20),
            #nn.ReLU(),
            #nn.Dropout(p=cfg.model.dropout),
            #nn.Linear(in_features=20, out_features=20)
        )

        self.correct = []
        self.count = []

        self.correct_test = []
        self.count_test = []

    def collate_fn(self,samples):
        #tuple to list

        x_seq = [sample[0] for sample in samples]
        seq_lengths = torch.LongTensor([sample[1] for sample in samples])
        x_creature = torch.stack([sample[2] for sample in samples], dim=0).view(-1,1)

        y_label = torch.stack([sample[3] for sample in samples], dim=0)

        #zero pad so entire batch has same length
        seq_tensor = Variable(torch.zeros((len(x_seq), seq_lengths.max()))).long()
        for idx, (seq, seq_len) in enumerate(zip(x_seq, seq_lengths)):
            seq_tensor[idx, :seq_len] = LongTensor(seq)

        #add creature label to each item in sequence
        x_creature = torch.cat([x_creature for i in range(seq_tensor.size(1))], dim=1)

        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        seq_tensor = seq_tensor[perm_idx]

        return seq_tensor, seq_lengths, x_creature, y_label


    def forward(self,x_seq,seq_lengths,x_creature):
        seq_tensor = self.embed_seq(x_seq)
        x_creature = self.embed_creature(x_creature)

        seq_tensor = torch.cat((seq_tensor, x_creature), dim=2)
        seq_tensor = pack_padded_sequence(seq_tensor, seq_lengths.cpu(), batch_first=True)
        _, self.hidden_state = self.gru(seq_tensor)#,torch.zeros(self.n_layers,128, self.hidden_dim, device=x.device))
        self.hidden_state = self.hidden_state.view(self.cfg.model.n_layers, self.num_directions,-1,self.cfg.model.hidden_size)
        out = torch.cat([self.hidden_state[-1,i,:,:] for i in range(self.hidden_state.size(1))], dim=1)
        out = self.fc_1(out)
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

    def train_dataloader(self):
        return DataLoader(Enzyme_Dataset(file_name=self.cfg.data.train.path,
                                           pre_pad=True,
                                           num_samples=self.cfg.data.train.num_samples),
                            collate_fn=self.collate_fn,
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
                           collate_fn=self.collate_fn,
                           shuffle=False,
                           num_workers=4,
                           pin_memory=True)

    def test_dataloader(self):
        return DataLoader(Enzyme_Dataset(file_name=self.cfg.data.eval.path,
                                         pre_pad=True,
                                         num_samples=self.cfg.data.eval.num_samples),
                          batch_size=self.cfg.data.eval.batch_size,
                          collate_fn=self.collate_fn,
                          shuffle=False,
                          num_workers=4,
                          pin_memory=True)

    def on_train_epoch_start(self):
        self.correct = []
        self.count = []
        self.hidden_state = 0



    def training_step(self, batch, batch_nb):
        x_seq, seq_lengths, x_creature, y_label = batch
        output = self.forward(x_seq, seq_lengths,x_creature)
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

    def validation_step(self, batch, batch_nb):
        x_seq, seq_lengths, x_creature, y_label = batch
        output = self.forward(x_seq, seq_lengths, x_creature)
        pred = output.detach().max(1, keepdim=True)[1]
        self.correct_test.append(pred.eq(y_label.view_as(pred)).sum().item())
        self.count_test.append(len(pred))
        loss = F.cross_entropy(input=output, target=y_label, reduction='sum')
        self.log('val_loss', loss)
        return {'val_loss': loss, 'log': {'val_loss': loss}}

    def on_validation_epoch_end(self):
        print("\n THE TEST ACCURACY FOR THIS EPOCH IS {} \n".format(sum(self.correct_test) / sum(self.count_test)))
        self.log('val_acc', sum(self.correct_test)/sum(self.count_test))

    def on_test_epoch_start(self):
        self.correct_test = []
        self.count_test = []
        self.hidden_state = 0

    def test_step(self, batch, batch_nb):
        x_seq, seq_lengths, x_creature, y_label = batch
        output = self.forward(x_seq, seq_lengths, x_creature)
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
                         fast_dev_run=False,
                         track_grad_norm=2,
                         gradient_clip_val=0.5,
                         gpus=1)#, early_stop_callback=None)
    trainer.fit(model)



if __name__ == '__main__':
    main()
