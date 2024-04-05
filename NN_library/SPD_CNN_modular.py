import matplotlib.pyplot as plt
import numpy as np

import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# query if we have GPU
dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Using device:', dev)



class SPD_Block(torch.nn.Module):
    def __init__(self, input_channels, out_size):
        super().__init__()

        self.out_size = out_size
        self.lower_tri_idx = np.tril_indices(self.out_size, k=0, m=None)
        self.diag_index = np.diag_indices(self.out_size, ndim=2)
        L_size = (self.out_size + 1) * self.out_size // 2

        self.convt = nn.ConvTranspose2d(input_channels, 1, kernel_size=(1, L_size), stride=(1, L_size), bias=False)
        

    def forward(self, x):
        b, _, n, _  = x.shape
        B_flat = self.convt(x).squeeze(1)
        B_mat = torch.zeros((b, n, self.out_size, self.out_size), device=dev)
        B_mat[:, :, self.lower_tri_idx[0], self.lower_tri_idx[1]] = B_flat
        B_mat[:, :, self.diag_index[0], self.diag_index[1]] = F.relu(B_mat[:, :, self.diag_index[0], self.diag_index[1]])
        B_mat = torch.matmul(B_mat, torch.transpose(B_mat, 2, 3))
        
        return B_mat



class Conv_block(torch.nn.Module):
    def __init__(self, w, kernel_size, input):
        super().__init__()
        
        pad = (kernel_size - 1) // 2
        self.c11 = nn.Conv2d(input, w, kernel_size=kernel_size, padding=pad, bias=True)
        self.c12 = nn.Conv2d(w, 2*w, kernel_size=1, bias=True)
        self.c13 = nn.Conv2d(2*w, 2*w, kernel_size=1, bias=True)

    def forward(self, x):
        c11 = F.relu(self.c11(x))
        c12 = F.relu(self.c12(c11))
        c13 = F.relu(self.c13(c12))
        
        return c13



class SPD_CNN(nn.Module):
    def __init__(self, res=5, kernels=[1, 3, 5], train_mode='stiffness', input=2):
        super().__init__()

        self.kernels = kernels
        self.train_mode = train_mode
        self.w = 16

        # Initialize the localization map
        glob_idx = []
        for i in kernels:
            glob_idx_i = np.reshape(np.arange((res+i)*(res+i), dtype=np.int16), (res+i, res+i))
            glob_idx.append(glob_idx_i)
        
        self.ix_1 = []
        self.ix_2 = []
        for i in range(len(kernels)):
            if kernels[i]!=1:
                pad = (kernels[i] - 1) // 2
                inner_node = glob_idx[i][pad:-pad, pad:-pad].flatten()
                inner_idx = np.vstack([2*inner_node, 2*inner_node+1]).T.flatten()
                ix_1, ix_2 = np.ix_(inner_idx, inner_idx)
                self.ix_1.append(torch.tensor(ix_1, dtype=torch.int).to(dev))
                self.ix_2.append(torch.tensor(ix_2, dtype=torch.int).to(dev))

        loc_map = []
        for j in range(len(kernels)):
            loc = np.zeros((res*res, 2*(kernels[j]+1)**2))
            for i in range(res*res):
                col = i % res
                row = i // res
                glob_view = glob_idx[j][row:row+kernels[j]+1, col:col+kernels[j]+1].flatten()
                loc[i] = np.vstack([2*glob_view, 2*glob_view+1]).T.flatten()
            loc_map.append(loc)

        self.loc_ix_1 = [[] for i in range(len(self.kernels))]
        self.loc_ix_2 = [[] for i in range(len(self.kernels))]
        for j in range(len(self.kernels)):
            for i in range(res*res):
                ix_1, ix_2 = np.ix_(loc_map[j][i], loc_map[j][i])
                self.loc_ix_1[j].append(torch.tensor(ix_1, dtype=torch.int).to(dev))
                self.loc_ix_2[j].append(torch.tensor(ix_2, dtype=torch.int).to(dev))
        
        # Convolution blocks
        self.conv_blocks = []
        for kernel in kernels:
            self.conv_blocks.append(Conv_block(self.w * 2**((kernel-1)//2), kernel, input))
        self.conv_blocks = nn.ModuleList(self.conv_blocks)

        # SPD transposed convolution blocks
        self.spd_blocks = []
        for kernel in kernels:
            self.FNN_blocks.append(SPD_Block(2*self.w * 2**((kernel-1)//2), 2*(kernel+1)**2))
        self.spd_blocks = nn.ModuleList(self.spd_blocks)


    def forward(self, x, zero_map, DBC, f):

        # Go through convolutional and SPD blocks
        c = []
        for i in range(len(self.kernels)):
            c.append(self.conv_blocks[i](x))

        b = []
        for i in range(len(self.kernels)):
            b.append(self.spd_blocks[i](c[i].view(-1, 2*self.w * 2**((self.kernels[i]-1)//2), x.shape[-2]*x.shape[-1], 1)))

        # Assemble global kernel matrices
        K = []
        for i in range(len(self.kernels)):
            K.append(torch.zeros(x.shape[0], 
                                 2*(x.shape[-2]+self.kernels[i])*(x.shape[-1]+self.kernels[i]), 
                                 2*(x.shape[-2]+self.kernels[i])*(x.shape[-1]+self.kernels[i]), device=dev))

        for j in range(len(self.kernels)):
            for i in range(x.shape[-2]*x.shape[-1]):
                K[j][:, self.loc_ix_1[j][i], self.loc_ix_2[j][i]] += b[j][:, i]

        for i in range(len(self.kernels)):
            if self.kernels[i]!=1:
                K[i] = K[i][:, self.ix_1[i-1], self.ix_2[i-1]]

        # Sum the global matrix and apply boundary conditions
        K_grid = K[0]
        for i in range(1, len(self.kernels)):
            K_grid += K[i]
        K_grid[zero_map] = 0

        # Find the matrix inverse
        if self.train_mode == 'compliance':
            K_BC = torch.zeros_like(K_grid)
            K_BC += K_grid + DBC
            C = torch.linalg.solve(K_BC, f)
            return K_grid, C
        
        else:
            return K_grid