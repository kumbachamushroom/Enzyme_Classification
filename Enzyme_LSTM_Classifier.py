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
        char_list = dict(enumerate(list(string.ascii_uppercase)))
        char2int = {ch: i + 1 for i, ch in char_list.items()}
        self.x_seq = list(map(lambda i: torch.tensor([char2int[ch]/26 for ch in i], dtype=torch.float32), self.x_seq))

        #convert labels to int
        self.x_creature = list(map(lambda label: torch.tensor(int(label[label.rfind('e')+1:])), self.x_creature))
        self.y_label = list(map(lambda label: torch.tensor(int(label[label.rfind('s')+1:])), self.y_label))

    def __len__(self):
        return len(self.y_label)

    def __getitem__(self, idx):
        # helper function to pre-pad amino acid sequence to be of equal lengths
        def zero_pad_sequence(sequence, length, pre=True):
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

        return zero_pad_sequence(sequence=self.x_seq[idx], length=self.maximum_length, pre=self.pre_pad), one_hot_encoding(self.x_creature[idx], 10), self.y_label[idx]#one_hot_encoding(self.y_label[idx],20)

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
        self.gru = nn.GRU(input_size=self.input_dim,hidden_size=self.hidden_dim, num_layers=self.n_layers, batch_first=cfg.model.bi, dropout=self.cfg.model.dropout, bidirectional=cfg.model.bi)
        # batch first = true: x-> batch_size, sequence_length, input_size
        # batch first = false: x -> seq_len, batch, input_size
        num_directions = 2 if cfg.model.bi else 1
        self.fc = nn.Linear(in_features=self.hidden_dim*num_directions+10, out_features=512+10)
        self.fc_2 = nn.Linear(in_features=512+10, out_features=256)
        self.fc_3 = nn.Linear(in_features=256, out_features=self.output_dim)
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm1d(num_features=512+10)
        self.batchnorm2 = nn.BatchNorm1d(num_features=256)
        self.correct = []
        self.count = []

        self.correct_test = []
        self.count_test = []
        self.hidden_state = 0
        self.save_hyperparameters()

    def forward(self,x_seq, x_creature):
        x_seq = torch.unsqueeze(x_seq,1)
        out, _ = self.gru(x_seq)#,torch.zeros(self.n_layers,128, self.hidden_dim, device=x.device))

        out = torch.cat((out[:,-1], x_creature), dim=1)
        out = self.relu(self.batchnorm(self.fc(out)))
        out = self.relu(self.batchnorm2(self.fc_2(out)))
        out = self.fc_3(out)
        return out

    def configure_optimizers(self):
        optimizer = optim.Adam(params=self.parameters(),
                               lr=self.cfg.model.lr_scheduler.initial_lr,
                               amsgrad=True)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                         factor=self.cfg.model.lr_scheduler.factor,
                                                         patience=self.cfg.model.lr_scheduler.patience,
                                                         min_lr=self.cfg.model.lr_scheduler.min_lr,
                                                         verbose=True)
        #lambda_lr= lambda epoch: ((1 / self.cfg.model.n_epochs) * epoch) if epoch < (0.5 * self.cfg.model.n_epochs) else ((1 / self.cfg.model.n_epochs) * self.cfg.model.n_epochs - epoch)
        #lambda_scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lambda_lr=lambda_lr)
        return {
                'optimizer' : optimizer,
                'lr_scheduler': scheduler,
                'monitor' : 'val_loss'
                }

    def train_dataloader(self):
        loader = DataLoader(dataset=Enzyme_Dataset(file_name=self.cfg.data.train.path,
                                                   pre_pad=True,
                                                   num_samples=self.cfg.data.train.num_samples),
                            batch_size=self.cfg.data.train.batch_size, shuffle=True,num_workers=2,
                            pin_memory=True, prefetch_factor=4)
        return loader

    def val_dataloader(self):
        loader = DataLoader(dataset=Enzyme_Dataset(file_name=self.cfg.data.eval.path,
                                                   pre_pad=True,
                                                   num_samples=self.cfg.data.eval.num_samples),
                            batch_size=self.cfg.data.eval.batch_size,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True)
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            Enzyme_Dataset(file_name=self.cfg.data.eval.path, pre_pad=True, num_samples=self.cfg.data.eval.num_samples),
            batch_size=self.cfg.data.eval.batch_size, shuffle=False, num_workers=4,pin_memory=True)
        return loader

    def on_train_epoch_start(self):
        self.correct = []
        self.count = []
        self.hidden_state = 0


    def training_step(self, batch, batch_nb):
        x_seq, x_creature, y_label = batch[0], batch[1], batch[2]
        #input = torch.cat((x_seq, x_creature),dim=1)
        output = self.forward(x_seq=x_seq, x_creature=x_creature)
        pred = output.detach().data.max(1, keepdim=True)[1]
        self.correct.append(pred.eq(y_label.view_as(pred)).sum().item())
        self.count.append(len(pred))
        loss = F.cross_entropy(input=output, target=y_label, reduction='mean')
        self.log('train_loss',loss)
        return {'loss':loss}

    def on_train_epoch_end(self, outputs):
        print("\n THE ACCURACY FOR THIS EPOCH IS {} \n".format(sum(self.correct)/sum(self.count)))
        self.log('train_acc',sum(self.correct)/sum(self.count))


    def on_validation_epoch_start(self):
        self.correct_test = []
        self.count_test = []
        self.hidden_state = 0

    def on_validation_epoch_end(self):
        print("\n THE TEST ACCURACY FOR THIS EPOCH IS {} \n".format(sum(self.correct_test) / sum(self.count_test)))
        self.log('val_acc', sum(self.correct_test)/sum(self.count_test))

    def validation_step(self, batch, batch_nb):
        x_seq, x_creature, y_label = batch[0], batch[1], batch[2]
        #output = self(torch.cat((x_seq, x_creature), dim=1))
        output = self.forward(x_seq=x_seq, x_creature=x_creature)
        pred = output.max(1, keepdim=True)[1]
        self.correct_test.append(pred.eq(y_label.view_as(pred)).sum().item())
        self.count_test.append(len(pred))
        loss = F.cross_entropy(input=output, target=y_label, reduction='mean')
        self.log('val_loss',loss)
        return {'val_loss': loss}

    def on_test_epoch_start(self):
        self.correct_test = []
        self.count_test = []
        self.hidden_state = 0

    def test_step(self, batch, batch_nb):
        x_seq, x_creature, y_label = batch[0], batch[1], batch[2]
        #output = self(torch.cat((x_seq, x_creature), dim=1))
        output = self.forward(x_seq=x_seq, x_creature=x_creature)
        pred = output.max(1, keepdim=True)[1]
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

    #def init_hidden(self, device):
    #    # h_0 of shape (num_layers * num_directions, batch, hidden_size): tensor containing the initial hidden state for each element in the batch.
    #    # Defaults to zero if not provided. If the RNN is bidirectional, num_directions should be 2, else it should be 1
    #    return torch.zeros(self.n_layers,self.cfg.data.batch_size, self.hidden_dim, device=device)

@hydra.main(config_path='/home/lucas/PycharmProjects/Enzyme_Classification/LSTM_only_config.yaml')
def main(cfg: DictConfig) -> None:
    max_epochs = cfg.model.n_epochs

    wandb_logger = WandbLogger(project='enzyme_classification')
    seed_everything(100)
    cuda = 1 if torch.cuda.is_available() else 0
    model = GRU_model(cfg)

    trainer = pl.Trainer(logger=wandb_logger if cfg.model.wandb else None,
                         default_root_dir='/home/lucas/PycharmProjects/Enzyme_Classification/models',
                         max_epochs=cfg.model.n_epochs,
                         fast_dev_run=False,
                         #track_grad_norm=2,
                         #gradient_clip_val=0.5,
                         gpus=1)#, early_stop_callback=None)
    trainer.fit(model)



if __name__ == '__main__':
    main()
