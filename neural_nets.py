import torch
import torch.nn as nn

class LSTM_embed(nn.Module):
    def __init__(self, z_dim, hidden_dim, vocab_size, device=torch.device('cpu')):
        super(LSTM_embed, self).__init__()
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.device = device
        self.num_layers = 1
        self.rxdx_embedding = nn.Embedding(vocab_size, hidden_dim).to(self.device)
        self.lstm = nn.LSTM(hidden_dim, z_dim, 1, batch_first=True).to(self.device)

    def forward(self, x):
        # x [batch size, sequence length]
        h0 = torch.zeros(self.num_layers, x.size(0), self.z_dim).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.z_dim).to(self.device)
        h1 = self.rxdx_embedding(x)  ## [batch size, sequence length, embedding dim]
        lstm_out, _ = self.lstm(h1, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        h2 = lstm_out[:, -1, :].to(self.device)  # [batch size, z dim]
        return h2


# class LSTM_iqvia(nn.Module):

    # def __init__(self, embedding_size, num_numerical_cols, output_size, layers, p=0.4, n_lstm=1, device=torch.device('cuda')):
    #     super().__init__()
    #     self.all_embeddings = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in embedding_size]).to(device)
    #     self.embedding_dropout = nn.Dropout(p)
    #     self.device = device
    #     self.batch_norm_num = nn.BatchNorm1d(num_numerical_cols).to(self.device)
    #     vocab_diag = 1536 + 1
    #     self.lstm = LSTM_embed(z_dim = 4, hidden_dim = 2, vocab_size=vocab_diag)

    #     all_layers = []
    #     num_categorical_cols = sum((nf for ni, nf in embedding_size))
    #     input_size = num_categorical_cols + num_numerical_cols + 4

    #     if n_lstm > 1:
    #         vocab_diag = 7896 + 1
    #         self.lstm = LSTM_embed(z_dim=4, hidden_dim=2, vocab_size=vocab_diag, device=device).to(self.device)
    #         vocab_prc = 880 + 1
    #         self.lstm_prc = LSTM_embed(z_dim=4, hidden_dim=2, vocab_size=vocab_prc, device=device).to(self.device)
    #         input_size += 4

    #     for i in layers:
    #         all_layers.append(nn.Linear(input_size, i))
    #         all_layers.append(nn.ReLU(inplace=True))
    #         all_layers.append(nn.BatchNorm1d(i))
    #         all_layers.append(nn.Dropout(p))
    #         input_size = i

    #     all_layers.append(nn.Linear(layers[-1], output_size))

    #     self.layers = nn.Sequential(*all_layers)

    # def forward(self, x):

    #     x_categorical = x[:,:3].type(torch.LongTensor).to(self.device)
    #     x_numerical = x[:,3:4].to(self.device)
    #     sequences = x[:,4:].type(torch.LongTensor).to(self.device)

    #     embeddings = torch.tensor([], device=self.device)
    #     for i,e in enumerate(self.all_embeddings):

    #         embeddings = torch.cat((embeddings,e(x_categorical[:,i].to(self.device))),1)

    #     x = embeddings
    #     x = self.embedding_dropout(x)

    #     x_numerical = self.batch_norm_num(x_numerical)
    #     x = torch.cat([x, x_numerical], 1)

    #     if sequences.shape[0]!=33:
    #         diag_sequences = sequences[:,:33].to(self.device)

    #         diag_sequences = self.lstm(diag_sequences)
    #         x = torch.cat([x,diag_sequences],1)

    #         prc_sequences = sequences[:,-33:].to(self.device)
    #         prc_sequences = self.lstm_prc(prc_sequences)

    #         x = torch.cat([x,prc_sequences],1)
    #         x = self.layers(x)

    #     else:
    #         sequences = self.lstm(sequences)
    #         x = torch.cat([x,sequences],1)
    #         x = self.layers(x)
    #     return x

class LSTM_iqvia_paper(nn.Module):

    def __init__(self, embedding_size, num_numerical_cols, output_size, layers, p=0.4, n_lstm=1, device=torch.device('cpu')):
        super().__init__()
        self.all_embeddings = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in embedding_size]).to(device)
        self.embedding_dropout = nn.Dropout(p)
        self.batch_norm_num = nn.BatchNorm1d(num_numerical_cols)
        vocab_diag = 941+1 #1536 + 1
        self.device = device
        self.lstm = LSTM_embed(z_dim = 4, hidden_dim = 2, vocab_size=vocab_diag, device=device)

        all_layers = []
        num_categorical_cols = sum((nf for ni, nf in embedding_size))
        input_size = num_categorical_cols + num_numerical_cols + 4

        for i in layers:
            all_layers.append(nn.Linear(input_size, i))
            all_layers.append(nn.ReLU(inplace=True))
            all_layers.append(nn.BatchNorm1d(i))
            all_layers.append(nn.Dropout(p))
            input_size = i

        all_layers.append(nn.Linear(layers[-1], output_size))

        self.layers = nn.Sequential(*all_layers)

    def forward(self, x):

        x_categorical = x[:,:2].type(torch.LongTensor).to(self.device)
        x_numerical = x[:,2:3].to(self.device)
        sequences = x[:,3:].type(torch.LongTensor).to(self.device)

        embeddings = torch.tensor([], device=self.device)
        for i,e in enumerate(self.all_embeddings):

            embeddings = torch.cat((embeddings,e(x_categorical[:,i].to(self.device))),1)
    
        x = embeddings
        x = self.embedding_dropout(x)
        x_numerical = self.batch_norm_num(x_numerical)
        x = torch.cat([x, x_numerical], 1)
 
        sequences = self.lstm(sequences)
        x = torch.cat([x,sequences],1)
        x = self.layers(x)
        return x








class LSTM_iqvia(nn.Module):

    def __init__(self, embedding_size, num_numerical_cols, output_size, layers, p=0.1, n_lstm=1, device=torch.device('cpu')):
        super().__init__()
        self.all_embeddings = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in embedding_size]).to(device)
        self.embedding_dropout = nn.Dropout(p)
        self.batch_norm_num = nn.BatchNorm1d(num_numerical_cols)
        vocab_diag = 941+1 
        self.device = device
        self.lstm = LSTM_embed(z_dim = 4, hidden_dim = 2, vocab_size=vocab_diag, device=device)

        all_layers = []
        num_categorical_cols = sum((nf for ni, nf in embedding_size))
        input_size = num_categorical_cols + num_numerical_cols + 4
        if n_lstm>1:
            vocab_diag = 7896 + 1
            self.lstm = LSTM_embed(z_dim = 4, hidden_dim=2, vocab_size=vocab_diag, device=device).to(self.device)
            vocab_prc = 880 + 1
            self.lstm_prc = LSTM_embed(z_dim = 4, hidden_dim=2, vocab_size=vocab_prc, device=device).to(self.device)
            input_size += 4


        for i in layers:
            all_layers.append(nn.Linear(input_size, i))
            all_layers.append(nn.ReLU(inplace=True))
            all_layers.append(nn.BatchNorm1d(i))
            all_layers.append(nn.Dropout(p))
            input_size = i

        all_layers.append(nn.Linear(layers[-1], output_size))

        self.layers = nn.Sequential(*all_layers)

    def forward(self, x):
        # print(x.is_c . .uda)
        # adhoc here
        # x_categorical = x[:,:3].type(torch.LongTensor).to(self.device)
        # x_numerical = x[:,3:4].to(self.device)
        x_categorical = x[:,:2].type(torch.LongTensor).to(self.device)
        x_numerical = x[:,2:3].to(self.device)
        sequences = x[:,3:].type(torch.LongTensor).to(self.device)

        embeddings = torch.tensor([], device=self.device)
        for i,e in enumerate(self.all_embeddings):
            # try:
            # print(embeddings.is_cuda)
            # print(e(x_categorical[:,i].to(self.device)).is_cuda)
            # print(self.device)
            embeddings = torch.cat((embeddings,e(x_categorical[:,i].to(self.device))),1)
            # embeddings.append(e(x_categorical[:,i]))
            # except:
            #     print('error here')
        # x = torch.cat(embeddings, 1)
        x = embeddings
        x = self.embedding_dropout(x)
        x_numerical = self.batch_norm_num(x_numerical)
        x = torch.cat([x, x_numerical], 1)

        sequences = self.lstm(sequences)
        x = torch.cat([x,sequences],1)
        x = self.layers(x)
        return x




class MLP_iqvia(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP_iqvia, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        # x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x











