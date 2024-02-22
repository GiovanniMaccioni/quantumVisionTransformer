import torch

import hybrid_mlp as H
import new_model as M
import data as D
import train as T

import os


def run(c):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if not os.path.exists(c['dir_path']):
        os.makedirs(c['dir_path'])

    with open(os.path.join(c['dir_path'], "config.txt"), 'w') as f:
        print(c, file=f)

    parameters = list(c.keys())
    
    if c['model'] == 'qvt':
        model = M.QuantumVisionTransformer(*[c[x] for x in parameters[7:]]).to(device)
    elif c['model'] == 'hmlp':
        model = H.HybridMLP(c['embed_dim'], c['num_layers'], c['num_classes'], c['hidden_dim'], c['vec_loader'], c['ort_layer']).to(device)
    else:
        model = M.QuantumVisionTransformer(*[c[x] for x in parameters[7:]]).to(device)
    

    optimizer = torch.optim.Adam(model.parameters(), lr=c['lr'])
    train_loader, test_loader = D.get_loaders(c['batch_size'], c['classes'])

    model.train()

    criterion = torch.nn.CrossEntropyLoss()

    #In this code (for now) I use the test set of PyTorch as the validation set. I don't use this to do any hyperparameter search or to compoare
    #any results
    T.train(model, c['epochs'], train_loader, test_loader, optimizer, criterion,c['dir_path'], device)

    return



