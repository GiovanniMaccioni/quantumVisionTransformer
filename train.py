import torch

import new_model as m
import data as d


if __name__=='__main__':

    #device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")

    config = dict(
        embed_dim = 4,
        hidden_dim = 32,
        num_heads = 8,
        num_layers = 1,
        patch_size = 14,#previous: 14
        num_channels = 1,
        num_patches = 4,#(dim_img/patch_size x dim_img/patch_size), previous: 4
        num_classes = 10,
        dropout = 0.2
    )
    quantum_model = m.QuantumVisionTransformer(**config).to(device)

    #TRAINING
    learning_rate=3e-3
    epochs=10


    optimizer = torch.optim.Adam(quantum_model.parameters(), lr=learning_rate, eps=1e-4)

    train_loader, test_loader = d.get_loaders()

    size = len(train_loader.dataset)

    quantum_model.train()

    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        #training
        for batch, (X, y) in enumerate(train_loader):
            # Compute prediction and loss
            #print(batch)
            X = X.to(device)
            y = y.to(device)
            pred = quantum_model(X)
            #print("pred: ", pred)
            loss = loss_fn(pred, y)#compute the loss 

            # Backpropagation
            optimizer.zero_grad()#zero the gradients for the optimizer
            loss.backward()#backwardpass
            #print(optimizer.param_groups)
            optimizer.step()#optimizer step
            #print the loss once in a while
            if batch % 5 == 0:
                loss_pok, current = loss.item(), batch * len(X)
                print(f"loss: {loss_pok:>7f}  [{current:>5d}/{size:>5d}]")
            #if (batch + 1) % 5 == 0:#DEBUG
            #    break
        break