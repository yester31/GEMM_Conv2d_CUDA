import torch # torch 1.9.0+cu111
import numpy as np


OC = 1
IN = 1
IC = 1
IH = 5
IW = 5
KH = 3
KW = 3

weight = torch.ones([OC, IC, KH, KW], dtype=torch.float32, requires_grad=False)
print(weight)

input_np = np.arange(1, IN * IC * IH * IW + 1).reshape(IN, IC, IH, IW)
input = torch.from_numpy(input_np).type(torch.FloatTensor)
print(input)

convolution = torch.nn.Conv2d(IC, OC, (KH, KH), stride=(1, 1), bias=False)
convolution.weight = torch.nn.Parameter(weight)

output = convolution(input)
print(output)

#output_c = np.fromfile("../output/C_Tensor", dtype=np.float32)
#output_py = output.detach().numpy().flatten()

#compare_two_tensor(output_py, output_c)
