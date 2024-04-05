import torch
import torch.nn as nn
import torch.optim as optim
from utilities.save_load import *

def Frobenius_norm(M, M_target, w=None):
    if w is None:
        diff = torch.linalg.matrix_norm(M - M_target, ord='fro', dim=(-2, -1))
        norm = torch.linalg.matrix_norm(M_target, ord='fro', dim=(-2, -1))
    else:
        W = torch.einsum("bi,bj->bij", w, w)
        diff = torch.linalg.matrix_norm(W * (M - M_target), ord='fro', dim=(-2, -1))
        norm = torch.linalg.matrix_norm(W * M_target, ord='fro', dim=(-2, -1))
    return diff / norm


def train(net, loaders, args):

    # network
    net.to(args['dev'])

    # optimizer
    optimizer = optim.Adam(net.parameters(), lr = args['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, threshold = 1e-10, verbose = True, eps=1e-10)

    n_train = len(loaders['train'].dataset)
    n_val = len(loaders['val'].dataset)

    losses_train = []
    losses_val = []

    for epoch in range(args['epochs']):

        # accumulate total loss over the training set
        L = 0
        net.train()
        if args['train_mode']=='stiffness':
            net.train_mode = 'stiffness'
            
        for i, (layout, K, C, zero_map, DBC, f) in enumerate(loaders['train']):
            layout = layout.to(args['dev'])
            K = K.to(args['dev'])
            C = C.to(args['dev'])
            f = f.to(args['dev'])
            DBC = DBC.to(args['dev'])
            K_net, C_net = net(layout, zero_map, DBC, f)
            l = (Frobenius_norm(K_net, K) + Frobenius_norm(C_net, C)).sum()
            loss_batch = l.detach().item()
            L += loss_batch
            optimizer.zero_grad()
            l.backward()
            optimizer.step() 


        # calculate the loss and accuracy of the validation set
        net.eval()
        L_val = 0

        if args['dev'] == "cuda":
                torch.cuda.empty_cache() 
        for j, (layout, K, C, zero_map, DBC, f) in enumerate(loaders['val']):
            layout = layout.to(args['dev'])
            K = K.to(args['dev'])
            C = C.to(args['dev'])
            f = f.to(args['dev'])
            DBC = DBC.to(args['dev'])
            if args['train_mode']=='stiffness':
                K_net = net(layout, zero_map, DBC, f)
                L_val += Frobenius_norm(K_net, K).sum().detach().cpu() + Frobenius_norm(K_net @ C, f).sum().detach().cpu()
            else:
                K_net, C_net = net(layout, zero_map, DBC, f) 
                L_val += (Frobenius_norm(K_net, K) + Frobenius_norm(C_net, C)).sum().detach().cpu()
        
        scheduler.step(L_val)   
        
        losses_train.append(L / n_train)
        losses_val.append(L_val / n_val)

        print(f'Epoch: {epoch} mean train loss: {L / n_train : .8e} mean val. loss: {L_val / n_val : .8e}')
        save_network(net, args['name'] + f'_{epoch}')        

    return losses_train, losses_val


