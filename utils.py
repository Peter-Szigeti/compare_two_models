import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn
import math

class Model():
    def __init__(self, mtype, n_input):
        if mtype == "sigmoid":
            self.sequential = torch.nn.Sequential(nn.Linear(n_input, 1), nn.Sigmoid())
        elif mtype == "linear":
            self.sequential = torch.nn.Sequential(nn.Linear(n_input, 1))
        
    def forward(self, x):
        return self.sequential(x)
    
    def train(self,X_train,y_train,lr=0.01,epochs=1000):
        loss_fn = nn.MSELoss()
        # learning rate decay
        optimizer = optim.SGD(self.sequential.parameters(),lr=lr)
        self.losses = []
        for i in range(epochs):
            y_h = self.forward(X_train)
            loss = loss_fn(y_h, y_train)
            self.losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    def calculate_R2(self, X_test, y_test):
        y_h = self.forward(X_test)
        return 1 - torch.sum((y_test - y_h)**2)/torch.sum((y_test - torch.mean(y_test))**2)


def choose_k_c_times(a,k,c):
    c = min(c, int(math.factorial(len(a))/(math.factorial(len(a)-k)*math.factorial(k))))
    chosen = []
    for i in range(c):
        while True:
            choice = tuple(np.random.choice(a, k, replace=False))
            if choice not in chosen: # here we should check other permutations of choice!!!
                chosen.append(choice)
                break
            if len(chosen) >= c:
                break
    return chosen

def generate_sigmoid_data(m, noise, minmax=0.8):
    # parameters of the logistic function
    L = 1
    w = 1
    b = 0
    g = lambda x: L / (1 + np.exp(-w * (x - b)))
    X = np.random.uniform(-minmax,minmax, (m, 1)) # 
    # y = 2 * x_0 + 0.5 * x_1 + 0.5 * x_2 + 1 * x_3 + 2 * x_4 + 4 * x_5
    y = g(2 * X[:, 0])
    y = y.reshape(-1, 1) # reshape y to be a column vector

    # add noise to X and y
    X += np.random.normal(0, noise, X.shape)
    y += np.random.normal(0, noise, y.shape)

    # # # normalize y and X between 0 and 1
    #y = (y - np.min(y)) / (np.max(y) - np.min(y))
    #X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0)) # converges faster when X is normalized(for some reason)
    # divide X and y into training and test sets
    X_train, X_test, y_train, y_test = X[:800], X[800:], y[:800], y[800:]
    return torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)


def generate_data(n_input, m):
    X = np.random.normal(0, 3, (m, n_input)) # Sample random integers from 0 to 10 from a Gaussian distribution into an array of shape (m, n_input)
    # y = 2 * x_0 + 0.5 * x_1 + 0.5 * x_2 + 1 * x_3 + 2 * x_4 + 4 * x_5
    y = 2 * X[:, 0] + 0.5 * X[:, 1]+ 0.5 *X[:, 2] + 1 * X[:, 3] + 2 * X[:, 4] + X[:, 5]
    y = y.reshape(-1, 1) # reshape y to be a column vector

    # add noise to X and y
    X += np.random.normal(0, 0.1, X.shape)
    y += np.random.normal(0, 0.1, y.shape)

    # # # normalize y and X between 0 and 1
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    #X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0)) # converges faster when X is normalized(for some reason)
    # divide X and y into training and test sets
    X_train, X_test, y_train, y_test = X[:800], X[800:], y[:800], y[800:]
    return torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)