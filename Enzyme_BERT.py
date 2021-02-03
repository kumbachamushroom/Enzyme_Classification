from nemo.collections import nlp as nemo_nlp
from nemo.utils.exp_manager import exp_manager

import os
import wget
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf


def main():
    config = OmegaConf.load(os.path.join(os.getcwd(), 'configs','text_classification_config.yaml'))
    #print(OmegaConf.to_yaml(config))
    config.model.dataset.num_classes = 20
    config.model.train_ds.file_path = os.path.join(os.getcwd(),'tsv_files','train.tsv')
    config.model.test_ds.file_path = os.path.join(os.getcwd(),'tsv_files','dev.tsv')
    config.model.validation_ds.path = os.path.join(os.getcwd(),'tsv_files','test.tsv')

    config.save_to = 'trained-model.nemo'
    config.export_to = 'trained-model.onnx'

    config.trainer.gpus = 1 if torch.cuda.is_available() else 0
    config.trainer.accelerator = None
    config.trainer.max_epochs = 100
    trainer = pl.Trainer(**config.trainer)

    #exp_dir = exp_manager(trainer=trainer, cfg=config.exp_manager)
    #print(exp_dir)

    config.model.language_model.pretrained_model_name = 'distilbert-base-cased'
    model = nemo_nlp.models.TextClassificationModel(cfg=config.model, trainer=trainer)

    #print(nemo_nlp.modules.get_pretrained_lm_models_list())

    trainer.fit(model)
    trainer.test(model)
    #model.save_to('/home/lucas/PycharmProjects/Enzyme_Classification/logs/trained-model.nemo')

    # extract the path of the best checkpoint from the training, you may update it to any checkpoint
    #checkpoint_path = trainer.checkpoint_callback.best_model_path
    #checkpoint_path = '/home/lucas/PycharmProjects/Enzyme_Classification/logs/trained-model.nemo'
    # Create an evaluation model and load the checkpoint
    #eval_model = nemo_nlp.models.TextClassificationModel.load_state_dict(, state_dict='/home/lucas/PycharmProjects/Enzyme_Classification/logs/trained-model.nemo')
    #eval_model = nemo_nlp.models.TextClassificationModel.load_from_checkpoint(checkpoint_path=checkpoint_path)

    # create a dataloader config for evaluation, the same data file provided in validation_ds is used here
    # file_path can get updated with any file
    #eval_config = OmegaConf.create(
    #    {'file_path': config.model.validation_ds.file_path, 'batch_size': 64, 'shuffle': False, 'num_samples': -1})
    #model.setup_test_data(test_data_config=eval_config)
    #eval_dataloader =model._create_dataloader_from_config(cfg=eval_config, mode='test')

    # a new trainer is created to show how to evaluate a checkpoint from an already trained model
    # create a copy of the trainer config and update it to be used for final evaluation
    #eval_trainer_cfg = config.trainer.copy()
    #eval_trainer_cfg.gpus = 1 if torch.cuda.is_available() else 0  # it is safer to perform evaluation on single GPU as PT is buggy with the last batch on multi-GPUs
    #eval_trainer_cfg.accelerator = None  # 'ddp' is buggy with test process in the current PT, it looks like it has been fixed in the latest master
    #eval_trainer = pl.Trainer(**eval_trainer_cfg)

    #eval_trainer.test(model=model, verbose=False)  # test_dataloaders=eval_dataloader

if __name__ == '__main__':
    main()