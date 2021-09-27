import numpy as np
import torch
import random
import pandas as pd
from torch.utils.data.dataset import Dataset
from ast import literal_eval
from tensorflow.keras.preprocessing.sequence import pad_sequences
from data_utils.data_module import CustomImageDataset
import warnings

warnings.filterwarnings("ignore")


# DATASETS
def get_iqvia():
    # data_dir = 'data/clean_large.csv'
    data_dir = './data'

    dataset_train = pd.read_csv('./data/paper_use_train_baseline.csv')
    dataset_test = pd.read_csv('./data/paper_use_test_baseline.csv')
    # time_series_columns = ['diag_code','prc_code','drug_code']
    time_series_columns = ['diag_code']
    categorical_columns = ['pag_gender','vacc_nm']#,'provider_state']
    numerical_columns = ['pat_age']
    label_columns = ['label']
    for category in categorical_columns:
        dataset_train[category] = dataset_train[category].astype('category')
        dataset_test[category] = dataset_test[category].astype('category')
    numerical_train_data = np.stack([dataset_train[col].values for col in numerical_columns], 1)
    numerical_test_data = np.stack([dataset_test[col].values for col in numerical_columns], 1)
    numerical_train_data = torch.tensor(numerical_train_data, dtype=torch.float)
    numerical_test_data = torch.tensor(numerical_test_data, dtype=torch.float)

    gender = dataset_train['pag_gender'].cat.codes.values
    brand = dataset_train['vacc_nm'].cat.codes.values
    categorical_train_data = np.stack([gender, brand], 1)
    categorical_train_data = torch.tensor(categorical_train_data, dtype=torch.int64)


    gender = dataset_test['pag_gender'].cat.codes.values
    brand = dataset_test['vacc_nm'].cat.codes.values
    categorical_test_data = np.stack([gender, brand], 1)
    categorical_test_data = torch.tensor(categorical_test_data, dtype=torch.int64)

    categorical_column_sizes = [len(
    set(list(dataset_train[column].cat.categories)+list(dataset_test[column].cat.categories))) for column in categorical_columns]
    # categorical_embedding_sizes = [(col_size, min(50, (col_size+1)//2)) for col_size in categorical_column_sizes]
    train_outputs = torch.tensor(dataset_train[label_columns].values).flatten()
    test_outputs = torch.tensor(dataset_test[label_columns].values).flatten()


    sentences = []
    for index, row in dataset_train.iterrows():
        sentences.append(literal_eval(row['diag_cd']))
        # sentences.append(torch.tensor(literal_eval(row['diag_code'])))
    # maxlen = int(np.mean([len(i) for i in sentences]))
    maxlen = 50
    time_series_train_data = pad_sequences(
        sentences, maxlen=maxlen, dtype='int32', padding='pre',
        truncating='pre', value=0.0
    )
    time_series_train_data = torch.tensor(time_series_train_data, dtype=torch.int64)

    sentences = []
    for index, row in dataset_test.iterrows():
        sentences.append(literal_eval(row['diag_cd']))
    maxlen = 50
    time_series_test_data = pad_sequences(
        sentences, maxlen=maxlen, dtype='int32', padding='pre',
        truncating='pre', value=0.0
    )
    time_series_test_data = torch.tensor(time_series_test_data, dtype=torch.int64)

    train_data = torch.cat((categorical_train_data, numerical_train_data, time_series_train_data), 1)
    test_data = torch.cat((categorical_test_data, numerical_test_data, time_series_test_data), 1)

    return train_data, train_outputs, test_data, test_outputs
    # dataset = pd.read_csv(data_dir, engine='python')
    # time_series_columns = ['diag_code']
    # categorical_columns = ['pag_gender', 'vacc_nm']
    # numerical_columns = ['pat_age']
    # label_columns = ['label']
    # # dataset['provider_zip'] = dataset['provider_zip'].fillna(0)
    # for category in categorical_columns:
    #     dataset[category] = dataset[category].astype('category')

    # numerical_data = np.stack([dataset[col].values for col in numerical_columns], 1)
    # numerical_data = torch.tensor(numerical_data, dtype=torch.float)

    # gender = dataset['pag_gender'].cat.codes.values
    # vcc_nm = dataset['vacc_nm'].cat.codes.values
    # zip = dataset['provider_zip'].cat.codes.values
    # categorical_data = np.stack([gender, vcc_nm], 1)
    # categorical_data = torch.tensor(categorical_data, dtype=torch.int64)

    # total_records = numerical_data.shape[0]
    # test_records = int(total_records * .2)

    # outputs = torch.tensor(dataset[label_columns].values).flatten()

    # time_sequences = []
    # for time_series in time_series_columns:
    #     sentences = []
    #     for index, row in dataset.iterrows():
    #         sentences.append(literal_eval(row[time_series]))
    #     maxlen = int(np.mean([len(i) for i in sentences]))
    #     sequences = pad_sequences(
    #         sentences, maxlen=maxlen, dtype='int32', padding='pre',
    #         truncating='pre', value=0.0
    #     )
    #     sequences = torch.tensor(sequences, dtype=torch.int64)
    #     time_sequences.append(sequences)
    # time_sequences_result = torch.cat(time_sequences, dim=1)

    # categorical_train_data = categorical_data[:total_records - test_records]
    # categorical_test_data = categorical_data[total_records - test_records:total_records]
    # numerical_train_data = numerical_data[:total_records - test_records]
    # numerical_test_data = numerical_data[total_records - test_records:total_records]
    # time_series_train_data = time_sequences_result[:total_records - test_records]
    # time_series_test_data = time_sequences_result[total_records - test_records:total_records]
    # train_outputs = outputs[:total_records - test_records]
    # test_outputs = outputs[total_records - test_records:total_records]

    # train_data = torch.cat((categorical_train_data, numerical_train_data, time_series_train_data), 1)
    # test_data = torch.cat((categorical_test_data, numerical_test_data, time_series_test_data), 1)
    return train_data, train_outputs, test_data, test_outputs


# SPLIT DATA AMONG CLIENTS

def split_data_by_state(data, labels, n_clients = 45, balancedness = None, method = None):
    n_data = data.shape[0]

    if balancedness >= 1.0:
        data_per_client = [n_data // n_clients] * n_clients
    else:
        fracs = balancedness ** np.linspace(0, n_clients - 1, n_clients)
        fracs /= np.sum(fracs)
        fracs = 0.1 / n_clients + (1 - 0.1) * fracs
        data_per_client = [np.floor(frac * n_data).astype('int') for frac in fracs]

        data_per_client = data_per_client[::-1]

    if sum(data_per_client) > n_data:
        print("Impossible Split")
        exit()
    # split data among clients
    flag = 0
    clients_split_by_state = []
    state_row_index_dict = np.load("state_row_index_f.npy", allow_pickle=True).item() # dict- key is state id, value is a list of row index
    for i in range(n_clients):
        # data_aug, labels_aug = data_augmentation(data[flag:flag + data_per_client[i], :],
        #                                  labels[flag:flag + data_per_client[i]], method)
 
        data_aug, labels_aug = data_augmentation(data[state_row_index_dict[i], :],
                                    labels[state_row_index_dict[i]], method)
        # i is client = state, data[[row_index],:]
        clients_split_by_state.append((data_aug, labels_aug))
        flag += data_per_client[i]

    return clients_split_by_state

#%%
#%%


# def split_data(data, labels, n_clients=20, balancedness=None, method=None):

#     n_data = data.shape[0]

#     if balancedness >= 1.0:
#         data_per_client = [n_data // n_clients] * n_clients
#     else:
#         fracs = balancedness ** np.linspace(0, n_clients - 1, n_clients)
#         fracs /= np.sum(fracs)
#         fracs = 0.1 / n_clients + (1 - 0.1) * fracs
#         data_per_client = [np.floor(frac * n_data).astype('int') for frac in fracs]

#         data_per_client = data_per_client[::-1]

#     if sum(data_per_client) > n_data:
#         print("Impossible Split")
#         exit()
#     # split data among clients
#     flag = 0
#     clients_split = []
#     for i in range(n_clients):
#         data_aug, labels_aug = data_augmentation(data[flag:flag + data_per_client[i], :],
#                                          labels[flag:flag + data_per_client[i]], method)
#         clients_split.append((data_aug, labels_aug))
#         flag += data_per_client[i]
#     return clients_split


def data_augmentation(data, labels, method):
    if method == 'down_sampling':
        pos_indexes = np.where(labels == 1)
        if len(pos_indexes[0]) == 0:
            return data, labels
        neg_indexes = np.delete(range(len(data)), pos_indexes)
        neg_indexes = np.random.choice(neg_indexes, len(pos_indexes[0]))
        data = torch.cat((data[pos_indexes], data[neg_indexes]))
        labels = torch.cat((labels[pos_indexes], labels[neg_indexes]))
        index = list(range(len(data)))
        random.shuffle(index)
        return data[index], labels[index]
    elif method == 'up_sampling':
        pos_indexes = np.where(labels == 1)
        if len(pos_indexes[0]) == 0:
            return data, labels
        times = (len(data) - len(pos_indexes[0])) // len(pos_indexes[0])
        for i in range(times - 1):
            data = torch.cat((data, data[pos_indexes]))
            labels = torch.cat((labels, labels[pos_indexes]))
        index = list(range(len(data)))
        random.shuffle(index)
        return data[index], labels[index]
    else:
        return data, labels


def get_data_loaders(hp):

    x_train, y_train, x_test, y_test = get_iqvia()

    split = split_data_by_state(x_train, y_train, n_clients=hp['n_clients'], balancedness=hp['balancedness'], method=hp["augmentation"])
    # split = split_data_by_state(x_train, y_train, n_clients=hp['n_clients'], balancedness=hp['balancedness'], method=hp["augmentation"])

    client_loaders = [torch.utils.data.DataLoader(CustomImageDataset(x, y),
                                                  batch_size=hp['batch_size'], shuffle=True) for x, y in split]
    train_loader = torch.utils.data.DataLoader(CustomImageDataset(x_train, y_train), batch_size=hp['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(CustomImageDataset(x_test, y_test), batch_size=hp['batch_size'], shuffle=False)

    stats = {"split": [x.shape[0] for x, y in split]}

    return client_loaders, train_loader, test_loader, stats


if __name__ == "__main__":

    hp = {'n_clients': 10, 'balancedness': 1.0, 'augmentation': 'up_sampling', 'batch_size': 64}
    x_train, y_train, x_test, y_test = get_data_loaders(hp)
    data_augmentation(x_test, y_test, method='up_sampling')
    split = split_data_by_state(x_train, y_train, balancedness=1.0)
