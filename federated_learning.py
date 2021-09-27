import json
import time
import argparse
import random
import torch
import os
from neural_nets import LSTM_iqvia, LSTM_iqvia_paper
import numpy as np
from distributed_training_utils import Client, Server
import experiment_manager as xpm
import default_hyperparameters as dhp
from tensorboardX import SummaryWriter
from data_utils.data_slover import get_data_loaders

random.seed(1023)
np.random.seed(1023)
torch.manual_seed(1023)
torch.cuda.manual_seed(1023)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument("--schedule", default="FL", type=str)
parser.add_argument("--start", default=0, type=int)
parser.add_argument("--end", default=None, type=int)
args = parser.parse_args()

print("Torch Version: ", torch.__version__)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# writer = SummaryWriter(comment=' FedProx with local epoch 5 beta 1')
writer = SummaryWriter(comment='per-FedAvg with local epoch 5 beta 1')

# Load the Hyperparameters of all Experiments to be performed and set up the Experiments
with open(os.path.join('config', 'federated_learning.json')) as data_file:
    experiments_raw = json.load(data_file)[args.schedule]

hp_dicts = [hp for x in experiments_raw for hp in xpm.get_all_hp_combinations(x)][args.start:args.end]
experiments = [xpm.Experiment(hyperparameters=hp) for hp in hp_dicts]


def run_experiments(experiments):
    print("Running {} Experiments..\n".format(len(experiments)))
    for xp_count, xp in enumerate(experiments):
        hp = dhp.get_hp(xp.hyperparameters)
        xp.prepare(hp)
        print(xp)

        # Load the Data and split it among the Clients
        client_loaders, train_loader, test_loader, stats = get_data_loaders(hp)

        # Instantiate Clients and Server with Neural Net
        # categorical_embedding_sizes = [(2, 1), (3, 2), (8801, 50)]
        categorical_embedding_sizes = [(2, 1), (3, 2)]
        n_lstm = 1
        net_model = LSTM_iqvia_paper(categorical_embedding_sizes, 1, 2, [100, 80, 60, 40, 20], p=0.1, n_lstm=n_lstm, device=device).to(device)

        clients = [Client(client_loader, None, net_model.to(device), hp, xp, i, algorithm=hp['algorithm'])
                   for i, client_loader in enumerate(client_loaders)]
        server = Server(None, test_loader, net_model.to(device), hp, xp, stats)

        # Start Distributed Training Process
        print("Start Distributed Training..")
        t1 = time.time()

        for c_round in range(1, hp['communication_rounds'] + 1):

            participating_clients = random.sample(clients, int(len(clients) * hp['participation_rate']))
            print("Starting Round {} training".format(c_round))

            # Clients do
            for client in participating_clients:
                client.synchronize_with_server(server)
                client.compute_weight_update(hp['local_iterations'])
                client.compress_weight_update_up(compression=hp['compression_up'], accumulate=hp['accumulation_up'],
                                                 count_bits=hp["count_bits"])

            # Server does
            server.aggregate_weight_updates(participating_clients, aggregation=hp['aggregation'])
            # server.compress_weight_update_down(compression=hp['compression_down'], accumulate=hp['accumulation_down'],
            #                                    count_bits=hp["count_bits"])

            print("Communication Round {} Finished".format(c_round))

            # Evaluate
            if xp.is_log_round(c_round):

                print("Experiment: {} ({}/{})".format(args.schedule, xp_count + 1, len(experiments)))
                print("Evaluate...")

                server.evaluate(writer=writer, iter=c_round)

                # Timing
                total_time = time.time() - t1
                avrg_time_per_c_round = total_time / c_round
                e = int(avrg_time_per_c_round * (hp['communication_rounds'] - c_round))
                print("Remaining Time (approx.):", '{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60),
                      "[{:.2f}%]\n".format(c_round / hp['communication_rounds'] * 100))

        # Delete objects to free up GPU memory
        del server
        clients.clear()
        torch.cuda.empty_cache()
        writer.close()


def print_optimizer(device):
    print("Optimizer:", device.hp['optimizer'])
    for key, value in device.optimizer.__dict__['defaults'].items():
        print(" -", key, ":", value)

    hp = device.hp
    base_batchsize = hp['batch_size']
    if hp['fix_batchsize']:
        client_batchsize = base_batchsize // hp['n_clients']
    else:
        client_batchsize = base_batchsize
    total_batchsize = client_batchsize * hp['n_clients']
    print(" - batchsize (/ total): {} (/ {})".format(client_batchsize, total_batchsize))
    print()


def print_model(device):
    print("Model {}:".format(device.hp['net']))
    n = 0
    for key, value in device.model.named_parameters():
        print(' -', '{:30}'.format(key), list(value.shape))
        n += value.numel()
    print("Total number of Parameters: ", n)
    print()


if __name__ == "__main__":
    run_experiments(experiments)
