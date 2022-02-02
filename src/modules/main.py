# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from transformer import Transformer


def trainIters(n_iters=15, batch_size=4, learning_rate=0.001):
    
    # X = np.load("data/X.npy")
    # y = np.load("data/Y.npy")
    # X = torch.from_numpy(X[0:80, :, 0:20]).float()
    # y = torch.from_numpy(y[0:80, 1:, 0:5]).float()
    # X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X = torch.rand(size=(80,7,20)) # batch,his_time,feature
    y = torch.rand(size=(80,6,5)) # batch,pred_time,tmc_code
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # print(X_train.type())
    # print(Y_train_.type())
    train_dataset = TensorDataset(torch.Tensor(X_train),torch.Tensor(Y_train))
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,drop_last=False,shuffle=True)
    test_dataset = TensorDataset(torch.Tensor(X_test),torch.Tensor(Y_test))
    test_dataloader = DataLoader(test_dataset,batch_size=batch_size,drop_last=False,shuffle=True)

    model = Transformer(feature_size=20, num_layers=2, dropout=0.3)
    # print(model)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    
    optimizer.zero_grad()
    for i in range(n_iters):
        model.train()
        train_losses = []
        target_length = 6

        for step, (x, y) in enumerate(train_dataloader):
            loss = 0
            optimizer.zero_grad()
            src,trg_out = (x, y) # [4,7,20] [4, 6, 5]

            for t in range(trg_out.size(1)):
                y_hat = model((src))
                loss += criterion(y_hat.view(-1),y[:,t,:].reshape(-1))
            
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item()/target_length)
       
        if i%5 == 0:
            model.eval()
            val_losses = []
            for step, (x, y) in enumerate(test_dataloader):
                loss_items = np.zeros(target_length)
                optimizer.zero_grad()
                src,trg_out = (x, y)
                
                for t in range(trg_out.size(1)):
                    y_hat = model((src))
                    loss_items[t] = criterion(y_hat.squeeze(1),y[:,t,:])
                val_losses.append(loss_items) 
            print("Epoch {}: training loss {} validation loss {}".format(i, np.array(train_losses).mean()**0.5,
                                                                         np.sqrt(np.stack(val_losses).mean(axis=0))))
        
