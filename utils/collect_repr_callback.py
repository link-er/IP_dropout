from pytorch_lightning import Callback
import numpy as np
from pathlib import Path

class CollectRepresentationDistribution(Callback):
    def __init__(self, train_saving_dataloader, test_dataloader, directory, freq=10):
        self.freq = freq
        self.train_saving_dataloader = train_saving_dataloader
        self.test_dataloader = test_dataloader
        self.directory = Path(directory)

    def on_train_epoch_end(self, *args, **kwargs):
        epoch = args[1].current_epoch
        saving_train_data = not (self.directory / "train_inputs").exists()
        if epoch % self.freq == 0:
            args[1].core.train()
            no_noise_train_representations = []
            train_inputs = []
            train_labels = []
            for b,l in self.train_saving_dataloader:
                b = b.cuda()
                no_noise_b_out = args[1].core.representation(b)
                no_noise_train_representations += no_noise_b_out.detach().cpu().numpy().tolist()
                if saving_train_data:
                    train_inputs += b.detach().cpu().numpy().tolist()
                    train_labels += l.detach().cpu().numpy().tolist()
            np.save(self.directory / ("no_noise_train_representations_"+str(epoch)), np.array(no_noise_train_representations))
            if saving_train_data:
                np.save(self.directory / "train_inputs", np.array(train_inputs))
                np.save(self.directory / "train_labels", np.array(train_labels))
            del no_noise_b_out
            del b
            del l

            saving_test_data = not (self.directory / "test_inputs").exists()
            no_noise_test_representations = []
            test_inputs = []
            test_labels = []
            for b,l in self.test_dataloader:
                b = b.cuda()
                no_noise_b_out = args[1].core.representation(b)
                no_noise_test_representations += no_noise_b_out.detach().cpu().numpy().tolist()
                if saving_test_data:
                    test_inputs += b.detach().cpu().numpy().tolist()
                    test_labels += l.detach().cpu().numpy().tolist()
            np.save(self.directory / ("no_noise_test_representations_"+str(epoch)), np.array(no_noise_test_representations))
            if saving_test_data:
                np.save(self.directory / "test_inputs", np.array(test_inputs))
                np.save(self.directory / "test_labels", np.array(test_labels))
            del no_noise_b_out
            del b
            del l
