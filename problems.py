import torch
import betty
from betty.problems import ImplicitProblem
from utils import compute_loss


class Multitask(ImplicitProblem):
    def __init__(self, *args, **kwargs):
        self.train_metric = kwargs.pop('train_metric')
        ImplicitProblem.__init__(self, *args, **kwargs)

    def training_step(self, batch):
        train_datas, train_targets = batch
        train_datas = [x.to(self.device) for x in train_datas] if type(train_datas) is list else train_datas.to(self.device)
        train_targets = {k: y.to(self.device) for k, y in train_targets.items()}
        
        if type(train_datas) is list:
            train_pred = [self.forward(train_data, t) for t, train_data in enumerate(train_datas)]
        else:
            train_pred = self.forward(train_datas)

        train_loss = [compute_loss(train_pred[t], train_targets[task_id], task_id) for t, task_id in enumerate(train_targets)]
        loss = self.auto_lambda(torch.stack(train_loss))

        self.train_metric.update_metric(train_pred, train_targets, train_loss)
        return loss


class Reweight(ImplicitProblem):
    def __init__(self, *args, **kwargs):
        self.train_tasks = kwargs.pop('train_tasks')
        self.pri_tasks = kwargs.pop('pri_tasks')
        ImplicitProblem.__init__(self, *args, **kwargs)

    def training_step(self, batch):
        val_datas, val_targets = batch
        val_datas = [x.to(self.device) for x in val_datas] if type(val_datas) is list else val_datas.to(self.device)
        val_targets = {k: y.to(self.device) for k, y in val_targets.items()}
        
        # define weighting for primary tasks (with binary weights)
        pri_weights = []
        for t in self.train_tasks:
            if t in self.pri_tasks:
                pri_weights += [1.0]
            else:
                pri_weights += [0.0]

        if type(val_datas) is list:
            val_pred = [self.multitask(val_data, t) for t, val_data in enumerate(val_datas)]
        else:
            val_pred = self.multitask(val_datas)
        # compute validation data loss on primary tasks
        val_loss = self.model_fit(val_pred, val_targets)
        loss = sum([w * val_loss[i] for i, w in enumerate(pri_weights)])
        return loss

    def model_fit(self, pred, targets):
        """
        define task specific losses
        """
        loss = [compute_loss(pred[i], targets[task_id], task_id) for i, task_id in enumerate(self.train_tasks)]
        return loss
