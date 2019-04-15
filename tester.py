"""
Author: Fayyaz Minhas, DCIS, PIEAS, PO Nilore, Islamabad, Pakistan
Email/web: http://faculty.pieas.edu.pk/fayyaz/

A barebones linear regression example with pytorch
Demonstrates: Representation, Evaluation and Optimization
as well as the concept of loss functions and automatic differentiation
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.modules as nn
#Let's generate some data
inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
targets = np.array([1,2,2,3])
device = torch.device('cpu')
# device = torch.device('cuda') # Uncomment this to run on GPU
x = torch.from_numpy(inputs).float()
y = torch.from_numpy(targets).float()
N, D_in,D_out = x.shape[0], x.shape[1], 1

# Create random Tensors for weights; setting requires_grad=True means that we
# want to compute gradients for these Tensors during the backward pass.
w = torch.randn(D_in+1, D_out, device=device, requires_grad=True)
#Note: we have added one additional weight (for bias)
lossf = nn.loss.MSELoss()#nn.loss.L1Loss()
learning_rate = 1e-1
L = [] #history of losses
for t in range(100):
  # Forward pass: compute predicted y using operations on Tensors. Since w1
  # has requires_grad=True, operations involving w1 will cause
  # PyTorch to build a computational graph, allowing automatic computation of
  # gradients. 
  """
  # REPRESENTATION
  """
  y_pred = x.mm(w[1:]).flatten()+w[0] #Implementing w'x+b

  """
  # EVALUATION
  """
  # Compute and print loss. Loss is a Tensor of shape (), and loss.item()
  # is a Python number giving its value.
  #loss = (y_pred - y).pow(2).mean() 
  loss = lossf(y_pred,y)
  obj = loss+0.0*w[1:].norm() #empirical loss + regularization
  L.append((loss.item(),obj.item())) #save for history
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
  with torch.no_grad():
    w -= learning_rate * w.grad
    # Manually zero the gradients after running the backward pass
    w.grad.zero_()

plt.plot(L)
plt.grid(); plt.xlabel('Epochs'); plt.ylabel('loss,obj')
print("Predictions: ",y_pred)