#!/usr/bin/env python

from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
import sys
import os
import time

TARGET_LEN = 500

# >>> from sklearn import tree
# >>> X = [[0, 0], [1, 1]]
# >>> Y = [0, 1]
# >>> clf = tree.DecisionTreeClassifier()
# >>> clf = clf.fit(X, Y)

def pad_or_truncate(some_list, target_len, rand_pad=False):
    if len(some_list) >= target_len:
        return some_list[:target_len]
    else:
        pad_len = target_len - len(some_list)
        new_list = some_list[:]
        if rand_pad:
            new_list.extend(os.urandom(pad_len))
        else:
            new_list.extend([0] * pad_len)
        return new_list

# read input
t0 = time.clock()
raw_data = []
raw_lengths = []
for line in sys.stdin:
    fields = line.strip().strip("[").strip("]").split("][")
    intfields = [int(x,16) for x in fields]
    raw_lengths.append(len(intfields))
    raw_data.append(pad_or_truncate(intfields, TARGET_LEN, rand_pad=True))

X = raw_data
Y = raw_lengths
t1 = time.clock()

print("Loaded %d samples ranging from %d-%d bytes in %f seconds" %
      (len(raw_data), min(raw_lengths), max(raw_lengths), t1-t0))

# clf = tree.DecisionTreeClassifier()
# scores = cross_val_score(clf, X, Y, cv=5)
# print("Decision Tree Classifier scores:", scores)

# clf.fit(X,Y)
# print "Done"

# Yprime = clf.predict(X)
# print Yprime
# for i in xrange(len(Yprime)):
#     if (Y[i] == Yprime[i]):
#         status = "CORRECT"
#     else:
#         status = "WRONG"
#     print(Y[i], Yprime[i], status)

# Try out AdaBoostClassifier
# adaboost_clf = AdaBoostClassifier(n_estimators=10)
# scores = cross_val_score(adaboost_clf, X, Y, cv=5)
# print("AdaBoost Classifier scores:", scores)

#############################################################################

# Try out neural network
print("Trying out a Neural Network!")
import numpy as np
np.set_printoptions(threshold=np.nan)

"""
PyTorch: nn
-----------

A fully-connected ReLU network with two hidden layers, trained to predict
y from x by minimizing squared Euclidean distance.

This implementation uses the nn package from PyTorch to build the network.
PyTorch autograd makes it easy to define computational graphs and take gradients,
but raw autograd can be a bit too low-level for defining complex neural networks;
this is where the nn package can help. The nn package defines a set of Modules,
which you can think of as a neural network layer that has produces output from
input and may have some trainable weights.
"""
import torch
from torch.autograd import Variable

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 4000, TARGET_LEN, 30, 1

# Create random Tensors to hold inputs and outputs, and wrap them in Variables.
# x = Variable(torch.randn(N, D_in))
# y = Variable(torch.randn(N, D_out), requires_grad=False)

# Create numpy arrays that hold normalized inputs and outputs, and wrap them in
# Variables.
x = Variable(torch.FloatTensor(X[:N]) / 255.0)
y = np.array(Y[:N], dtype=np.float32)
y.reshape((-1,1))
y = y / TARGET_LEN
y = Variable(torch.from_numpy(y), requires_grad=False)

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Variables for its weight and bias.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, D_out),
)

optimizer = torch.optim.SGD(model.parameters(), lr = 1e-5, momentum=0.9,
                            weight_decay=1e-2)

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(size_average=False)

#learning_rate = 1e-5
for t in range(3000000): #35000 steps??
    optimizer.zero_grad()

    # Forward pass: compute predicted y by passing x to the model. Module objects
    # override the __call__ operator so you can call them like functions. When
    # doing so you pass a Variable of input data to the Module and it produces
    # a Variable of output data.
    y_pred = model(x)

    # Compute and print loss. We pass Variables containing the predicted and true
    # values of y, and the loss function returns a Variable containing the
    # loss.
    loss = loss_fn(y_pred, y)
    print("SGDIteration:", t, "\tMeanSquaredError =", loss.data[0])

    # Zero the gradients before running the backward pass.
    #model.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Variables with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    loss.backward()

    # Update the weights using gradient descent. Each parameter is a Variable, so
    # we can access its data and gradients like we did before.
    #for param in model.parameters():
    #    param.data -= learning_rate * param.grad.data
    optimizer.step()

# Test on training results:
print("-"*70)
print("Training Accuracy")
print("-"*70)
y_pred_denormalized = torch.round(y_pred.data * float(TARGET_LEN)).numpy()
y_denormalized = torch.round(y.data * float(TARGET_LEN)).view((-1,1)).numpy()
y_correct = (y_pred_denormalized == y_denormalized)
print("shape of y_pred:", y_pred_denormalized.shape)
print("shape of y:", y_denormalized.shape)
print("shape of y_correct:", y_correct.shape)
print(np.hstack((y_pred_denormalized, y_denormalized, y_correct)))
train_correct = int(np.sum(y_correct))
train_total = y_correct.shape[0]
print("%d correct out of %d (%2f %%)" %
      (train_correct, train_total, float(train_correct)/train_total*100.0))

# Test accuracy (only if there is data to test against)
if (len(X) <= N):
    sys.exit(0)

print("-"*70)
print("Testing Accuracy")
print("-"*70)
x = Variable(torch.FloatTensor(X[N:]) / 255.0, requires_grad=False)
y = np.array(Y[N:], dtype=np.float32)
y.reshape((-1,1))
y = y / TARGET_LEN
y = Variable(torch.from_numpy(y), requires_grad=False)

y_pred = model(x)

y_pred_denormalized = torch.round(y_pred.data * float(TARGET_LEN)).numpy()
y_denormalized = torch.round(y.data * float(TARGET_LEN)).view((-1,1)).numpy()
y_correct = (y_pred_denormalized == y_denormalized)
print("shape of y_pred:", y_pred_denormalized.shape)
print("shape of y:", y_denormalized.shape)
print("shape of y_correct:", y_correct.shape)
print(np.hstack((y_pred_denormalized, y_denormalized, y_correct)))
test_correct = int(np.sum(y_correct))
test_total = y_correct.shape[0]
print("%d correct out of %d (%2f %%)" %
      (test_correct, test_total, float(test_correct)/test_total*100.0))



# Test on training:
# for i, x_vec in enumerate(X):
#     y_prediction = round(model(torch.FloatTensor(x_vec) / 255.0) * TARGET_LEN)
#     y_actual = Y[i]
#     if y_prediction == y_actual:
#         result = "CORRECT"
#     else:
#         result = "WRONG"
#     print("item #%d: predicted = %d, actual = %d, %s" \
#           % (i, y_prediction, y_actual, result))
