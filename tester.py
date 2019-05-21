"""
Author: Fayyaz Minhas, DCIS, PIEAS, PO Nilore, Islamabad, Pakistan
Email/web: http://faculty.pieas.edu.pk/fayyaz/

A barebones single neuron model example with pytorch
You will need plotit for this (https://github.com/foxtrotmike/plotit)
Demonstrates: 
    Representation, Evaluation and Optimization
    concept of loss functions, SRM objective function
    automatic differentiation
    optimization
Things to try: What happens if you:
    change the input from bipolar to binary
    change the targets from bipolar to binary
    change the loss function
    change the activation function
    change the regularization parameter
    can solve a linearly inseparable classification problem
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.modules as nn
import torch.optim as optim

#Let's generate some data
inputs = 2*np.array([[0,0],[0,1],[1,0],[1,1]],dtype=np.float)-1
targets = 2*np.array([1,0,0,0],dtype=np.float)-1
device = torch.device('cpu')
#device = torch.device('cuda') # Uncomment this to run on GPU
x = torch.from_numpy(inputs).float()
y = torch.from_numpy(targets).float()
N, D_in,D_out = x.shape[0], x.shape[1], 1
# Loss function implementation
def hinge(y_true, y_pred):
    zero = torch.Tensor([0]) 
    return torch.mean(torch.max(zero, 1 - y_true * y_pred))
# Other loss functions can also be implemented
#lossf = nn.MSELoss()#nn.L1Loss()
# Create random Tensors for weights; setting requires_grad=True means that we
# want to compute gradients for these Tensors during the backward pass.
wb = torch.randn(D_in+1, D_out, device=device, requires_grad=True)
#Note: we have added one additional weight (for bias)
learning_rate = 1e-1
optimizer = optim.Adam([wb], lr=learning_rate)
L = [] #history of losses
for t in range(100):
  # Forward pass: compute predicted y using operations on Tensors. Since w1
  # has requires_grad=True, operations involving w1 will cause
  # PyTorch to build a computational graph, allowing automatic computation of
  # gradients. 
  """
  # REPRESENTATION
  """
  w = wb[1:]
  b = wb[0]
  net_in = x.mm(w).flatten()+b #Implementing w'x+b
  y_pred = net_in
  #y_pred = torch.tanh(net_in) #uncomment this to apply tanh activation
  """
  # EVALUATION
  """
  # Compute and print loss. Loss is a Tensor of shape (), and loss.item()
  # is a Python number giving its value.
  #loss = (y_pred - y).pow(2).mean() #loss = lossf(y_pred,y)
  loss = hinge(y,y_pred)  
  obj = loss+0.0*torch.transpose(w,1,0).mm(w) #empirical loss + regularization
  L.append((loss.item(),obj.item())) #save for history and plotting
  """
  #OPTIMIZATION
  """
  # Use autograd to compute the backward pass. This call will compute the
  # gradient of loss with respect to all Tensors with requires_grad=True.
  # After this call w1.grad will be Tensors holding the gradient
  # of the loss with respect to w1.
  obj.backward()

  # Update weights using gradient descent. For this step we just want to mutate
  # the values of w1 in-place; we don't want to build up a computational
  # graph for the update steps, so we use the torch.no_grad() context manager
  # to prevent PyTorch from building a computational graph for the updates
  """
  # Manually update weights using gradient descent
  with torch.no_grad():
    wb = wb - learning_rate * wb.grad #gradient descent update
    # Manually zero the gradients after running the backward pass
    wb.grad.zero_()
  """
  # Using built-in optimizer
  optimizer.step()
  optimizer.zero_grad()


wbn = wb.detach().numpy()
plt.close("all")
plt.plot(L)
plt.grid(); plt.xlabel('Epochs'); plt.ylabel('value');plt.legend(['Loss','Objective'])
print("Predictions: ",y_pred)
print("Weights: ",wbn)
plt.figure()
def clf(inputs): 
  return inputs@wbn[1:]+wbn[0]

from plotit import plotit
plotit(inputs,targets,clf=clf, conts=[-1,0,1])
