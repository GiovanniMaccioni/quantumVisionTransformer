import torch
import numpy as np

def img_to_patch(x, patch_size, flatten_channels=True):
    """
    Inputs:
        x - torch.Tensor representing the image of shape [B, C, H, W]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H//patch_size, patch_size, W//patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5) # [B, H', W', C, p_H, p_W]
    x = x.flatten(1,2)              # [B, H'*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(2,4)          # [B, H'*W', C*p_H*p_W]
    return x

def get_RBS_parameters(x):
    # get recursively the angles
    def angles(y):
        d = y.shape[-1]
        if d == 2:
            #print(y.shape)
            thetas = torch.acos(y[:,:, 0] / torch.linalg.norm(y, ord=None, dim=2))
            #print("thetas.shape: ", thetas.shape)
            signs = (y[:, :, 1] > 0.).int()
            thetas = signs * thetas + (1. - signs) * (2. * np.pi - thetas)
            #print("thetas.shape: ", thetas.shape)
            thetas = thetas[:,:, None]
            return thetas
        else:
            thetas = torch.acos(torch.linalg.norm(y[:,:, :d//2], ord=None, dim=2, keepdim=True) / torch.linalg.norm(y, ord=None, dim=2, keepdim=True))
            #print("else: thetas.shape: ", thetas.shape)
            #print("y[:,:, :d // 2]", y[:, :, :d // 2])
            thetas_l = angles(y[:, :, :d//2])
            thetas_r = angles(y[:, :, d//2 :])
            thetas = torch.cat((thetas, thetas_l, thetas_r), axis=2)
            #print("thetas.shape: ", thetas.shape)
        return thetas

    # result
    thetas = angles(x)

    return torch.nan_to_num(thetas)