import torch
import torch.optim as optim
import compression_utils as comp
import torch.nn.functional as F
from optimizer_utils.MySGD import MySGD
from metric_utils.evaluation import evaluate

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def copy(target, source):
    for name in target:
        target[name].data = source[name].data.clone()


def copy_decoder(target, source):
    for name in target:
        if "down_convs" not in name:
            target[name].data = source[name].data.clone()


def add(target, source):
    for name in target:
        target[name].data += source[name].data.clone()


def scale(target, scaling):
    for name in target:
        target[name].data = scaling * target[name].data.clone()


def subtract(target, source):
    for name in target:
        target[name].data -= source[name].data.clone()


def subtract_(target, minuend, subtrahend):
    for name in target:
        target[name].data = minuend[name].data.clone() - subtrahend[name].data.clone()


def add_subtract(target, minuend, subtrahend):
    for name in target:
        target[name].data = target[name].data + minuend[name].data.clone() - subtrahend[name].data.clone()


def average(target, sources):
    for name in target:
        target[name].data = torch.mean(torch.stack([source[name].data for source in sources]), dim=0).clone()
        

def average_decoder(target, sources):
    for name in target:
        if "down_convs" not in name:
            target[name].data = torch.mean(torch.stack([source[name].data for source in sources]), dim=0).clone()


def weighted_average(target, sources, weights):
    for name in target:
        summ = torch.sum(weights)
        n = len(sources)
        modify = [weight / summ * n for weight in weights]
        target[name].data = torch.mean(torch.stack([m * source[name].data for source, m in zip(sources, modify)]),
                                       dim=0).clone()

def weighted_average_decoder(target, sources, weights):
    for name in target:
        if "down_convs" not in name:
            summ = torch.sum(weights)
            n = len(sources)
            modify = [weight / summ * n for weight in weights]
            target[name].data = torch.mean(torch.stack([m * source[name].data for source, m in zip(sources, modify)]),
                                           dim=0).clone()


def weighted_weights(target, source, alpha=0.25):
    for name in target:
        target[name].data = alpha * target[name].data.clone() + (1 - alpha) * source[name].data.clone()


def majority_vote(target, sources, lr):
    for name in target:
        mask = torch.stack([source[name].data.sign() for source in sources]).sum(dim=0).sign()
        target[name].data = (lr * mask).clone()


def compress(target, source, compress_fun):
    """compress_fun : a function f : tensor (shape) -> tensor (shape)"""
    for name in target:
        target[name].data = compress_fun(source[name].data.clone())


def cal_proximal_term(minuend, subtrahend):
    proximal_term = 0.0
    for name in minuend:
        proximal_term += (minuend[name].data.clone() - subtrahend[name].data.clone()).norm(2)
    return proximal_term


class DistributedTrainingDevice(object):
    def __init__(self, train_loader, test_loader, model, hyperparameters, experiment):
        self.hp = hyperparameters
        self.xp = experiment
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model


class Client(DistributedTrainingDevice):

    def __init__(self, train_loader, val_loader, model, hyperparameters, experiment, num_id, algorithm="FedAvg"):
        super().__init__(train_loader, val_loader, model, hyperparameters, experiment)

        self.id = num_id

        # Parameters
        self.W = {name: value for name, value in self.model.named_parameters()}
        self.W_old = {name: torch.zeros(value.shape).to(device) for name, value in self.W.items()}
        self.dW = {name: torch.zeros(value.shape).to(device) for name, value in self.W.items()}
        self.dW_compressed = {name: torch.zeros(value.shape).to(device) for name, value in self.W.items()}
        self.A = {name: torch.zeros(value.shape).to(device) for name, value in self.W.items()}
        self.n_params = sum([T.numel() for T in self.W.values()])
        self.bits_sent = []

        # Optimizer (specified in self.hp, initialized using the suitable parameters from self.hp)
        # optimizer_object = getattr(optim, self.hp['optimizer'])
        # self.optimizer_parameters = {k: v for k, v in self.hp.items() if k in optimizer_object.__init__.__code__.co_varnames}

        # self.optimizer = optimizer_object(self.model.parameters(), **self.optimizer_parameters)

        self.algorithm = algorithm
        if self.algorithm == 'per_FedAvg':
            self.optimizer = MySGD(self.model.parameters(), self.hp['lr'])
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), self.hp['lr'])

        # State
        self.epoch = 0
        self.train_loss = 0.0


    def synchronize_with_server(self, server):

        # W_client = W_server
        copy(target=self.W, source=server.W)


    def train_cnn(self, iterations, mu=0.1, algorithm='FedAvg'):

        running_loss = 0.0
        w = torch.Tensor([1.0, 1.0]).to(device)

        for i in range(iterations):

            self.epoch += 1

            for j, (x, y) in enumerate(self.train_loader):
                if x.shape[0] <= 1:
                    continue
                # x shape: N * C * H * W, y shape N * H * W
                x, y = x.to(device), y.to(device).long()

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                y_= self.model(x)

                loss = F.cross_entropy(y_, y, w)

                if algorithm == 'FedProx':
                    proximal_term = cal_proximal_term(self.W, self.W_old)
                    loss += mu * proximal_term / 2

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

        running_loss /= len(self.train_loader)
        return running_loss / iterations

    def per_FedAvg(self, iterations, beta):
        running_loss = 0.0
        # w = torch.Tensor([1.0, 16.0]).to(device)
        temp_W = {name: torch.zeros(value.shape).to(device) for name, value in self.W.items()}
        for i in range(iterations):
            train_loader = iter(self.train_loader)
            self.epoch += 1
            copy(temp_W, self.W)

            x, y = next(train_loader)
            x, y = x.to(device), y.to(device).long()
            self.optimizer.zero_grad()
            y_ = self.model(x)
            loss = F.cross_entropy(y_, y)
            loss.backward()
            self.optimizer.step()

            # x, y = next(train_loader)
            # x, y = x.to(device), y.to(device).long()
            # self.optimizer.zero_grad()
            # y_ = self.model(x)
            # loss = F.cross_entropy(y_, y, w)
            # loss.backward()

            # restore the model parameters to the one before first update
            copy(self.W, temp_W)
            self.optimizer.step(beta=beta)

            running_loss += loss

        return running_loss / iterations

    def compute_weight_update(self, iterations=1):

        # Training mode
        self.model.train()

        # W_old = W
        copy(target=self.W_old, source=self.W)

        if self.algorithm == 'FedAvg' or self.algorithm == 'FedProx':
            # W = SGD(W, D)
            self.train_loss = self.train_cnn(iterations, algorithm = self.algorithm)
        elif self.algorithm == 'per_FedAvg':
            self.train_loss = self.per_FedAvg(iterations, beta=0.05)

        print("Training loss at epoch {} of Client {} is {:3f}".format(self.epoch, self.id, self.train_loss))

        # dW = W - W_old
        subtract_(target=self.dW, minuend=self.W, subtrahend=self.W_old)

    def compress_weight_update_up(self, compression=None, accumulate=False, count_bits=False):

        if accumulate and compression[0] != "none":
            # compression with error accumulation
            add(target=self.A, source=self.dW)
            compress(target=self.dW_compressed, source=self.A, compress_fun=comp.compression_function(*compression))
            subtract(target=self.A, source=self.dW_compressed)

        else:
            # compression without error accumulation
            compress(target=self.dW_compressed, source=self.dW, compress_fun=comp.compression_function(*compression))

        if count_bits:
            # Compute the update size
            self.bits_sent += [comp.get_update_size(self.dW_compressed, compression)]


class Server(DistributedTrainingDevice):

    def __init__(self, train_loader, val_loader, model, hyperparameters, experiment, stats):
        super().__init__(train_loader, val_loader, model, hyperparameters, experiment)

        # Parameters
        self.W = {name: value for name, value in self.model.named_parameters()}
        self.dW_compressed = {name: torch.zeros(value.shape).to(device) for name, value in self.W.items()}
        self.dW = {name: torch.zeros(value.shape).to(device) for name, value in self.W.items()}

        self.A = {name: torch.zeros(value.shape).to(device) for name, value in self.W.items()}

        self.n_params = sum([T.numel() for T in self.W.values()])
        self.bits_sent = []

        self.client_sizes = torch.Tensor(stats["split"]).cuda()


    def aggregate_weight_updates(self, clients, aggregation="mean"):

        # dW = aggregate(dW_i, i=1,..,n)
        if aggregation == "mean":
            average(target=self.dW, sources=[client.dW_compressed for client in clients])

        elif aggregation == "weighted_mean":
            weighted_average(target=self.dW, sources=[client.dW_compressed for client in clients],
                             weights=torch.stack([self.client_sizes[client.id] for client in clients]))

        elif aggregation == "majority":
            majority_vote(target=self.dW, sources=[client.dW_compressed for client in clients], lr=self.hp["lr"])

    def compress_weight_update_down(self, compression=None, accumulate=False, count_bits=False):
        if accumulate and compression[0] != "none":
            # compression with error accumulation
            add(target=self.A, source=self.dW)
            compress(target=self.dW_compressed, source=self.A, compress_fun=comp.compression_function(*compression))
            subtract(target=self.A, source=self.dW_compressed)

        else:
            # compression without error accumulation
            compress(target=self.dW_compressed, source=self.dW, compress_fun=comp.compression_function(*compression))

        add(target=self.W, source=self.dW_compressed)

        if count_bits:
            # Compute the update size
            self.bits_sent += [comp.get_update_size(self.dW_compressed, compression)]

    def evaluate(self, writer, iter):

        self.model.eval()
        test_loss = 0
        pred_list = []
        label_list = []
        prob_list = []
        for idx, (data, target) in enumerate(self.test_loader):
            data, target = data.to(device), target.to(device)
            log_probs = self.model(data)
            test_loss += F.cross_entropy(log_probs, target).item()
            y_prob, y_pred = log_probs.data.max(1, keepdim=True)
            pred_list.append(y_pred.cpu())
            prob_list.append(y_prob.cpu())
            label_list.append(target.cpu())

        pred_all = pred_list[0]
        prob_all = prob_list[0]
        label_all = label_list[0]
        for i in range(len(pred_list) - 1):
            pred_all = torch.cat((pred_all, pred_list[i + 1]), dim=0)
            prob_all = torch.cat((prob_all, prob_list[i + 1]), dim=0)
            label_all = torch.cat((label_all, label_list[i + 1]), dim=0)

        accuracy, preci, recal, mcc, f1, kappa, roc_auc, pr_auc, matrix = evaluate(label_all, pred_all, prob_all)
        if writer is not None:
            writer.add_scalar('Val/Accuracy', accuracy, iter)
            writer.add_scalar('Val/Precision', preci, iter)
            writer.add_scalar('Val/Recall', recal, iter)
            writer.add_scalar('Val/MCC', mcc, iter)
            writer.add_scalar('Val/f1', f1, iter)
            writer.add_scalar('Val/kappa', kappa, iter)
            writer.add_scalar('Val/roc_auc', roc_auc, iter)
            writer.add_scalar('Val/pr_auc', pr_auc, iter)


