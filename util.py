import numpy as np
import torch

# V, I, SOC, T, Vgap
# Min-max scale with (0.01, 0.99) quantile
PANLI_NORMALIZER = np.array([[803., -337., 0., 25., 0.], 
                             [972., 510., 90., 38., 0.03]])

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def point_adjustment(gt, pr):
    flag = False
    pa = pr.copy()
    for i in range(len(gt)):
        if flag:
            if gt[i] == 1:
                pa[i] = 1
            else:
                flag = False
        else:
            if gt[i] == 1:
                flag = True
    return pa
