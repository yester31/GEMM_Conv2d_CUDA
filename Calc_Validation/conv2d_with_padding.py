import torch # torch 1.9.0+cu111
import numpy as np
from compare import *

OC = 3
IN = 2
IC = 2
IH = 4
IW = 4
KH = 3
KW = 3
TP = 1
BP = 2
LP = 3
RP = 4

weight = torch.ones([OC, IC, KH, KW], dtype=torch.float32, requires_grad=False)
print(weight)

input_np = np.arange(1, IN * IC * IH * IW + 1).reshape(IN, IC, IH, IW)
input = torch.from_numpy(input_np).type(torch.FloatTensor)
print(input)

p2d = (LP, RP, TP, BP)
input_padded = torch.nn.functional.pad(input, p2d, "constant", 0)
print(input_padded)

conservertive_convolution = torch.nn.Conv2d(IC, OC, (KH, KH), stride=(1, 1), bias=False)
conservertive_convolution.weight = torch.nn.Parameter(weight)

output = conservertive_convolution(input_padded)
print(output)

output_c = np.fromfile("../output/C_Tensor_zp", dtype=np.float32)
output_py = output.detach().numpy().flatten()

compare_two_tensor(output_py, output_c)