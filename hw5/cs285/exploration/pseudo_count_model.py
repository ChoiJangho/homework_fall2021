from cs285.infrastructure import pytorch_util as ptu
from .base_exploration_model import BaseExplorationModel
import torch.optim as optim
from torch import nn
import torch
import numpy as np

class PseudoCountModel(nn.Module, BaseExplorationModel):
    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)
        self.ob_dim = hparams['ob_dim']
        self.n_layers = hparams['rnd_n_layers']
        self.size = hparams['rnd_size']
        self.optimizer_spec = optimizer_spec
        self.loss_fun = nn.BCELoss()

        self.classifier = ptu.build_mlp(self.ob_dim, 1, self.n_layers, self.size, output_activation='sigmoid')

        self.optimizer = self.optimizer_spec.constructor(
            self.classifier.parameters(),
            **self.optimizer_spec.optim_kwargs
        )
        self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            self.optimizer_spec.learning_rate_schedule,
        )

        self.classifier.to(ptu.device)

    def forward(self, ob_no):
        # <DONE>: Get the prediction error for ob_no
        # HINT: Remember to detach the output of self.f!
        prob_novel = self.classifier(ob_no)
        density = (1 - prob_novel) / prob_novel
        return density

    def forward_np(self, ob_no):
        ob_no = ptu.from_numpy(ob_no)
        return ptu.to_numpy(self(ob_no))

    def update(self, ob_no, ob_reb):
        novel_data_size = ob_no.shape[0]
        replay_data_size = ob_reb.shape[0]
        labels = ptu.from_numpy(np.expand_dims(np.concatenate([np.ones(novel_data_size), np.zeros(replay_data_size)]), 1).astype(
            np.float32))
        prob_novel = self.classifier(ptu.from_numpy(np.concatenate([ob_no, ob_reb])))
        loss = self.loss_fun(prob_novel, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
