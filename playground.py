import torch
import torch.nn as nn
        
X = torch.tensor([[1, 1], [2, 3], [4, 3], [5, 5]], dtype=torch.float32)
print(nn.init.xavier_normal_(X))
print(nn.init.kaiming_normal_(X))

import deeplib
import deeplib.nn as nn
import numpy as np

X = deeplib.Tensor([[1, 1], [2, 3], [4, 3], [5, 5]], dtype=np.float32)
print(nn.init.xavier_normal_(X))

print(nn.init.kaiming_normal_(X))
