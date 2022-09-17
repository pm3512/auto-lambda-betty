import argparse
from audioop import mul

import torch.optim as optim
import torch.utils.data.sampler as sampler

from auto_lambda import AutoLambda
from create_network import *
from create_dataset import *
from utils import *
import problems

import betty
from betty.configs import Config, EngineConfig
from betty.engine import Engine

from datetime import datetime

parser = argparse.ArgumentParser(description='Multi-task/Auxiliary Learning: CIFAR-100')
parser.add_argument('--mode', default='none', type=str)
parser.add_argument('--port', default='none', type=str)

parser.add_argument('--weight', default='equal', type=str, help='multi-task weighting: equal, dwa, uncert, autol')
parser.add_argument('--gpu', default=0, type=int, help='gpu ID')
parser.add_argument('--autol_init', default=0.1, type=float, help='initialisation for auto-lambda')
parser.add_argument('--autol_lr', default=3e-4, type=float, help='learning rate for auto-lambda')
parser.add_argument('--subset_id', default=0, type=int, help='domain id for cifar-100, -1 for MTL mode')
parser.add_argument('--seed', default=0, type=int, help='random seed ID')

opt = parser.parse_args()

torch.manual_seed(opt.seed)
np.random.seed(opt.seed)
random.seed(opt.seed)

# create logging folder to store training weights and losses
if not os.path.exists('logging'):
    os.makedirs('logging')
time_str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

# define model, optimiser and scheduler
device = torch.device("cuda:{}".format(opt.gpu) if torch.cuda.is_available() else "cpu")
model = MTLVGG16(num_tasks=20).to(device)
train_tasks = {'class_{}'.format(i): 5 for i in range(20)}
pri_tasks = {'class_{}'.format(opt.subset_id): 5} if opt.subset_id >= 0 else train_tasks

total_epoch = 200

if opt.weight == 'autol':
    params = model.parameters()
    meta_weight_ls = np.zeros([total_epoch, len(train_tasks)], dtype=np.float32)

elif opt.weight in ['dwa', 'equal']:
    T = 2.0  # temperature used in dwa
    lambda_weight = np.ones([total_epoch, len(train_tasks)], dtype=np.float32)
    params = model.parameters()

elif opt.weight == 'uncert':
    logsigma = torch.tensor([-0.7] * len(train_tasks), requires_grad=True, device=device)
    params = list(model.parameters()) + [logsigma]
    logsigma_ls = np.zeros([total_epoch, len(train_tasks)], dtype=np.float32)

optimizer = optim.SGD(params, lr=0.1, weight_decay=5e-4, momentum=0.9)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epoch)


# define dataset
trans_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
])

trans_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
])

train_sets = CIFAR100Multiclass(root='dataset', train=True, transform=trans_train)
if opt.subset_id == -1:
    test_sets = CIFAR100Multiclass(root='dataset', train=False, transform=trans_test)
else:
    test_sets = CIFAR100MTL(root='dataset', train=False, transform=trans_test, subset_id=opt.subset_id)

batch_size = 32

train_loaders = torch.utils.data.DataLoader(dataset=train_sets, batch_size=batch_size, shuffle=True, num_workers=2)
    

# a copy of train_loader with different data order, used for Auto-Lambda meta-update
if opt.weight == 'autol':
    val_loaders = torch.utils.data.DataLoader(dataset=train_sets, batch_size=batch_size, shuffle=True, num_workers=2)

test_loaders = torch.utils.data.DataLoader(dataset=test_sets, batch_size=batch_size, shuffle=False, num_workers=2)


# Train and evaluate multi-task network
if opt.subset_id >= 0:
    print('CIFAR-100 | Training Task: All Domains | Primary Task: {} in Multi-task / Auxiliary Learning Mode with VGG-16'
          .format(test_sets.subset_class.title()))
else:
    print('CIFAR-100 | Training Task: All Domains | Primary Task: All Domains in Multi-task / Auxiliary Learning Mode with VGG16')

print('Applying Multi-task Methods: Weighting-based: {}'
      .format(opt.weight.title()))

train_batch = len(train_loaders)
test_batch = len(test_loaders)

train_metric = TaskMetric(train_tasks, pri_tasks, batch_size, total_epoch, 'cifar100')
if opt.subset_id >= 0:
    test_metric = TaskMetric(train_tasks, pri_tasks, batch_size, total_epoch, 'cifar100')
else:
    test_metric = TaskMetric(train_tasks, pri_tasks, batch_size, total_epoch, 'cifar100', include_mtl=True)

if opt.weight == 'autol':
    multitask_config = Config(type='darts', unroll_steps=1)
    reweight_config = Config(type='darts', unroll_steps=1, retain_graph=True)

    # lower level task
    multitask = problems.Multitask(
        name='multitask',
        module=model,
        optimizer=optimizer,
        train_data_loader=train_loaders,
        config=multitask_config,
        device=device,
        train_metric=train_metric
    )

    # loss weighting linear layer
    meta_module = nn.Linear(len(train_tasks), 1, bias=False, device=device)
    meta_module.weight.data.fill_(opt.autol_init)

    meta_optimizer = optim.Adam(meta_module.parameters(), lr=opt.autol_lr)

    # upper level task
    auto_lambda = problems.Reweight(
        name='auto_lambda',
        config=reweight_config,
        train_tasks=train_tasks,
        pri_tasks=pri_tasks,
        device=device,
        optimizer=meta_optimizer,
        module=meta_module,
        train_data_loader=val_loaders
    )

    engine_config = EngineConfig(
        train_iters=train_batch * total_epoch,
        valid_step=train_batch
    )

    u2l = {auto_lambda: [multitask]}
    l2u = {multitask: [auto_lambda]}
    dependencies = {'u2l': u2l, 'l2u': l2u}
    problems = [auto_lambda, multitask]

    class AutoLambdaEngine(Engine):
        def __init__(self, config, problems, scheduler_epoch, dependencies=None, env=None):
            super().__init__(config, problems, dependencies, env)
            # workaround to perform scheduler step once per epoch instead of every batch
            self.scheduler_epoch = scheduler_epoch
        def validation(self):
            self.scheduler_epoch.step()
            epoch = test_metric.epoch_counter
            train_str = self.multitask.train_metric.compute_metric(only_pri=True)
            self.multitask.train_metric.reset()

            # evaluating test data
            with torch.no_grad():
                if opt.subset_id >= 0:
                    test_dataset = iter(test_loaders)
                    for _ in range(test_batch):
                        test_data, test_target = test_dataset.next()
                        test_data = test_data.to(device)
                        test_target = test_target.to(device)

                        test_pred = self.multitask(test_data, opt.subset_id)
                        test_loss = F.cross_entropy(test_pred, test_target)

                        test_metric.update_metric([test_pred], {'class_{}'.format(opt.subset_id): test_target}, [test_loss])
                else:
                    test_datasets = iter(test_loaders)
                    for _ in range(test_batch):
                        test_datas, test_targets = test_datasets.next()
                        test_datas = [x.to(device) for x in test_datas]
                        test_targets = {k: y.to(device) for k, y in test_targets.items()}

                        test_pred = [self.multitask(test_data, t) for t, test_data in enumerate(test_datas)]
                        test_loss = [compute_loss(test_pred[t], test_targets[task_id], task_id) for t, task_id in enumerate(test_targets)]

                        test_metric.update_metric(test_pred, test_targets, test_loss)

            test_str = test_metric.compute_metric(only_pri=True)
            test_metric.reset()
            if opt.subset_id >= 0:
                print('Epoch {:04d} | TRAIN:{} || TEST:{} | Best: {} {:.4f}'
                .format(epoch, train_str, test_str, test_sets.subset_class.title(),
                        test_metric.get_best_performance('class_{}'.format(opt.subset_id))))
            else:
                print('Epoch {:04d} | TRAIN:{} || TEST:{} | Best: All {:.4f}'
                    .format(epoch, train_str, test_str, test_metric.get_best_performance('all')))

            meta_weight_ls[epoch] = self.auto_lambda.module.weight.data.detach().cpu()
            dict = {'train_loss': train_metric.metric, 'test_loss': test_metric.metric,
                    'weight': meta_weight_ls}

            class_names = train_sets.sets[0].class_dict.keys()
            print(get_weight_str_ranked(meta_weight_ls[epoch], list(class_names), 4))
            np.save('logging/mtl_cifar_{}_{}_{}_betty_{}.npy'.format(
                opt.subset_id, opt.weight, opt.seed, time_str
            ), dict)
        
    engine = AutoLambdaEngine(
        config=engine_config,
        problems=problems,
        dependencies=dependencies,
        scheduler_epoch=scheduler
    )

    engine.run()
    
else:
    for index in range(total_epoch):

        # apply Dynamic Weight Average
        if opt.weight == 'dwa':
            if index == 0 or index == 1:
                lambda_weight[index, :] = 1.0
            else:
                w = []
                for i, t in enumerate(train_tasks):
                    w += [train_metric.metric[t][index - 1, 0] / train_metric.metric[t][index - 2, 0]]
                w = torch.softmax(torch.tensor(w) / T, dim=0)
                lambda_weight[index] = len(train_tasks) * w.numpy()

        # evaluating train data
        model.train()
        train_datasets = iter(train_loaders)
        for k in range(train_batch):
            train_datas, train_targets = next(train_datasets)
            train_datas = [x.to(device) for x in train_datas]
            train_targets = {k: y.to(device) for k, y in train_targets.items()}

            optimizer.zero_grad()

            train_pred = [model(train_data, t) for t, train_data in enumerate(train_datas)]
            train_loss = [compute_loss(train_pred[t], train_targets[task_id], task_id) for t, task_id in enumerate(train_targets)]

            if opt.weight in ['equal', 'dwa']:
                loss = sum(w * train_loss[i] for i, w in enumerate(lambda_weight[index]))

            if opt.weight == 'uncert':
                loss = sum(1 / (2 * torch.exp(w)) * train_loss[i] + w / 2 for i, w in enumerate(logsigma))

            loss.backward()
            optimizer.step()

            train_metric.update_metric(train_pred, train_targets, train_loss)

        train_str = train_metric.compute_metric(only_pri=True)
        train_metric.reset()

        # evaluating test data
        model.eval()
        with torch.no_grad():
            test_dataset = iter(test_loaders)
            if opt.subset_id >= 0:
                for k in range(test_batch):
                    test_data, test_target = test_dataset.next()
                    test_data = test_data.to(device)
                    test_target = test_target.to(device)

                    test_pred = model(test_data, opt.subset_id)
                    test_loss = F.cross_entropy(test_pred, test_target)

                    test_metric.update_metric([test_pred], {'class_{}'.format(opt.subset_id): test_target}, [test_loss])
            else:
                for k in range(test_batch):
                    test_datas, test_targets = test_dataset.next()
                    test_datas = [x.to(device) for x in test_datas]
                    test_targets = {k: y.to(device) for k, y in test_targets.items()}

                    test_pred = [model(test_data, t) for t, test_data in enumerate(test_datas)]
                    test_loss = [compute_loss(test_pred[t], test_targets[task_id], task_id) for t, task_id in enumerate(test_targets)]
                    test_metric.update_metric(test_pred, test_targets, test_loss)

        test_str = test_metric.compute_metric(only_pri=True)
        test_metric.reset()

        scheduler.step()
        class_names = train_sets.sets[0].class_dict.keys()

        if opt.subset_id >= 0:
            print('Epoch {:04d} | TRAIN:{} || TEST:{} | Best: {} {:.4f}'
            .format(index, train_str, test_str, test_sets.subset_class.title(),
                    test_metric.get_best_performance('class_{}'.format(opt.subset_id))))
        else:
            print('Epoch {:04d} | TRAIN:{} || TEST:{} | Best: All {:.4f}'
                .format(index, train_str, test_str, test_metric.get_best_performance('all')))

        if opt.weight in ['dwa', 'equal']:
            dict = {'train_loss': train_metric.metric, 'test_loss': test_metric.metric,
                    'weight': lambda_weight}
            print(get_weight_str_ranked(lambda_weight[index], list(class_names), 4))

        if opt.weight == 'uncert':
            logsigma_ls[index] = logsigma.detach().cpu()
            dict = {'train_loss': train_metric.metric, 'test_loss': test_metric.metric,
                    'weight': logsigma_ls}
            print(get_weight_str_ranked(1 / (2 * np.exp(logsigma_ls[index])), list(class_names), 4))

        np.save('logging/mtl_cifar_{}_{}_{}_betty_{}.npy'.format(
            opt.subset_id, opt.weight, opt.seed, time_str
        ), dict)
