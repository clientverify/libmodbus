#!/usr/bin/env python

from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import sys
import os
import time
import argparse
from collections import OrderedDict

np.set_printoptions(threshold=np.nan)

description_text = \
"""\
Learn libmodbus packet lengths.

The idea is to predict a packet's length based on its content -- i.e.,
see if a machine can implicitly learn the "type" and "length" fields of
a packet.

  y = f(x)

Task: given a bunch of (x, y) pairs, try to learn f.

x = Sequence of bytes of a packet, as an L-dimensional feature vector
    (where L = 500 is a reasonable default for libmodbus because its
    packets max out at slightly more than 256 bytes). If the
    packetlength < L, pad with random bytes; if packetlength > L,
    truncate.

y = Length of x before padding or truncation.
"""

def main():
    global args
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=description_text)

    parser.add_argument('layersizes', metavar='L', type=int, nargs='+',
                        help=
"""Sequence of units per layer, including input and
output, where the last number (output) must be 1 and
the first number is the number of input features.
Example: 500 30 10 1""")
    parser.add_argument('-b', '--batchsize', metavar='B', type=int,
                        default=500, help="batch size (default 500)")
    parser.add_argument('-e', '--epochs', metavar='E', type=int,
                        default=100, help="number of epochs (default 100)")
    parser.add_argument('-t', '--trainsize', metavar='M', type=int,
                        default=3000,
                        help="training data size (#examples, default 3000)")
    parser.add_argument('-l', '--learnrate', metavar='L', type=float,
                        default=1e-5, help="learning rate (default 1e-5)")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="be more chatty on stderr")
    parser.add_argument('-V', '--veryverbose', action='store_true',
                        help="be extremely chatty, show MSE every iteration")
    args = parser.parse_args()

    INPUT_FEATURES = args.layersizes[0] # input dimension
    if args.layersizes[-1] != 1:
        print("Number of units per layer: ", args.layersizes)
        parser.error("The final layer must be a single unit, but got %d." %
                     args.layersizes[-1])
    if args.veryverbose: # veryverbose implies verbose
        args.verbose = True

    # Read input
    X, Y = read_packet_lines(sys.stdin, INPUT_FEATURES)

    # Split into training and dev sets
    if len(X) <= args.trainsize:
        eprint("Error: not enough data for %d training examples" %
               args.trainsize)
        return -1
    trainX = X[:args.trainsize]
    trainY = Y[:args.trainsize]
    devX = X[args.trainsize:]
    devY = Y[args.trainsize:]
    if args.verbose:
        eprint("Data split into %d training and %d dev examples" %
               (len(trainX), len(devX)))

    # Train a model
    model = nn_train(trainX, trainY, args)

    # Run the model on the training set
    trainY_predict = run_model(model, trainX)
    print("-"*70)
    print("Test-on-Train Accuracy")
    print("-"*70)
    accuracy = compute_accuracy(trainY, trainY_predict, trainX)

    # Run the model on the dev set
    devY_predict = run_model(model, devX)
    print("-"*70)
    print("Hold-out Cross Validation ('Dev set') Accuracy")
    print("-"*70)
    accuracy = compute_accuracy(devY, devY_predict, devX)

    return 0

def pad_or_truncate(some_list, target_len, rand_pad=True):
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

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def read_packet_lines(f, num_features):
    t0 = time.clock()
    raw_data = []
    raw_lengths = []
    for line in f:
        fields = line.strip().strip("[").strip("]").split("][")
        intfields = [int(x,16) for x in fields]
        raw_lengths.append(len(intfields))
        raw_data.append(pad_or_truncate(intfields, num_features))

    X = raw_data
    Y = raw_lengths
    t1 = time.clock()
    if args.verbose:
        eprint("Loaded %d samples ranging from %d-%d bytes in %f seconds" %
               (len(raw_data), min(raw_lengths), max(raw_lengths), t1-t0))
    return [X, Y]

def normalize_tensor(X, Xmean=None, Xstd=None):
    # CAVEAT: watch out for NaN values!
    if Xmean is None:
        Xmean = X.mean(dim=0, keepdim=True)[0]
    if Xstd is None:
        Xstd = X.std(dim=0, keepdim=True)[0]
        Xstd = Xstd.numpy()
        Xstd[Xstd == 0] = 1.0 # don't divide by zero!
        Xstd = torch.FloatTensor(Xstd)
    Xnormalized = (X - Xmean)/Xstd
    return [Xnormalized, Xmean, Xstd]

def denormalize_tensor(Xnormalized, Xmean, Xstd):
    X = (Xnormalized) * Xstd + Xmean
    return X

def any_isnan(x):
    # use the fact that NaN != NaN
    return (x != x).any()

def nn_train(X, Y, params):
    """PyTorch: nn

    A fully-connected ReLU network with several hidden layers, trained to
    predict y from x by minimizing squared Euclidean distance.

    This implementation uses the nn package from PyTorch to build the network.
    PyTorch autograd makes it easy to define computational graphs and take
    gradients, but raw autograd can be a bit too low-level for defining complex
    neural networks; this is where the nn package can help. The nn package
    defines a set of Modules, which you can think of as a neural network layer
    that has produces output from input and may have some trainable weights.
    """
    N = params.batchsize
    # input and output dimensions; the others are hidden dimensions
    D_in = params.layersizes[0]
    D_out = params.layersizes[-1]

    # Use the nn package to define our model as a sequence of
    # layers. nn.Sequential is a Module which contains other Modules, and
    # applies them in sequence to produce its output. Each Linear Module
    # computes output from input using a linear function, and holds internal
    # Variables for its weight and bias.
    modules = OrderedDict()
    dims = params.layersizes
    for i in range(len(dims) - 2):
        modules["linear" + str(i+1)] = torch.nn.Linear(dims[i], dims[i+1])
        modules["relu" + str(i+1)] = torch.nn.ReLU()
    modules["linear" + str(len(dims)-1)] = torch.nn.Linear(dims[-2], dims[-1])
    nn_model = torch.nn.Sequential(modules)
    if args.verbose:
        eprint("Created PyTorch neural net architecture: " + str(nn_model))

    # Initialize weights randomly
    def init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)
    nn_model.apply(init_weights)

    # Create numpy arrays that hold normalized inputs and outputs, and put them
    # into a DataLoader
    Xnorm, Xmean, Xstd = normalize_tensor(torch.FloatTensor(X))
    assert not any_isnan(Xnorm)
    Y_np = np.array(Y, dtype=np.float32)
    Y_np.reshape((-1,1))
    Ynorm, Ymean, Ystd = normalize_tensor(torch.FloatTensor(Y_np).unsqueeze(1))
    assert not any_isnan(Ynorm)
    training_loader = DataLoader(TensorDataset(Xnorm, Ynorm),
                                 batch_size=N, num_workers=8)
    if args.verbose:
        eprint("X type and shape:", type(Xnorm), str(Xnorm.shape))
        eprint("Y type and shape:", type(Ynorm), str(Ynorm.shape))
        eprint("Batch size:", N)

    # Set optimizer
    optimizer = torch.optim.SGD(nn_model.parameters(),
                                lr = params.learnrate,
                                momentum=0.9,
                                weight_decay=1e-2)

    # The nn package also contains definitions of popular loss functions; in
    # this case we will use Mean Squared Error (MSE) as our loss function.
    loss_fn = torch.nn.MSELoss(size_average=False)

    # Train the neural network
    for epoch in range(params.epochs):
        for t, (Xbatch, Ybatch) in enumerate(training_loader):
            optimizer.zero_grad()

            # Wrap batch of training data in Variables
            x = Variable(Xbatch)
            y = Variable(Ybatch, requires_grad=False)

            # Forward pass: compute predicted y by passing x to the
            # model. Module objects override the __call__ operator so you can
            # call them like functions. When doing so you pass a Variable of
            # input data to the Module and it produces a Variable of output
            # data.
            y_pred = nn_model(x)

            # Compute and print loss. We pass Variables containing the predicted
            # and true values of y, and the loss function returns a Variable
            # containing the loss.
            loss = loss_fn(y_pred, y)
            if args.veryverbose or (t == params.trainsize/params.batchsize - 1):
                eprint("Epoch:", epoch+1,
                       "\tIteration: %2d" % (t+1),
                       "\tMeanSquaredError =", loss.data[0])

            # Backward pass: compute gradient of the loss with respect to all
            # the learnable parameters of the model. Internally, the parameters
            # of each Module are stored in Variables with requires_grad=True, so
            # this call will compute gradients for all learnable parameters in
            # the model.
            loss.backward()

            # Update the weights using gradient descent.
            optimizer.step()

    # Collect the trained model and return it
    model = {}
    model["nn_model"] = nn_model
    model["Xmean"] = Xmean
    model["Xstd"] = Xstd
    model["Ymean"] = Ymean
    model["Ystd"] = Ystd

    return model


def run_model(model, X):
    X = torch.FloatTensor(X)
    Xnorm = normalize_tensor(X, model["Xmean"], model["Xstd"])[0]
    xnorm = Variable(Xnorm, requires_grad=False)
    ynorm_pred = model["nn_model"](xnorm)
    Y_predict = denormalize_tensor(ynorm_pred.data,
                                   model["Ymean"],
                                   model["Ystd"])
    return Y_predict

def compute_accuracy(Y, Ypred, X=None):
    Y = np.reshape(np.asarray(Y), (-1, 1))
    Ypred = torch.round(Ypred).numpy()
    Ycorrect = (Y == Ypred)
    if args.verbose:
        eprint("shape of Ypred", Ypred.shape)
        eprint("shape of Y", Y.shape)
        eprint("First 10 items: [predicted, actual, isCorrect]")
        eprint(np.hstack((Ypred, Y, Ycorrect))[:10])
    test_correct = int(np.sum(Ycorrect))
    test_total = Ycorrect.shape[0]
    print("%d correct out of %d (%2f %%)" %
          (test_correct, test_total, float(test_correct)/test_total*100.0))

###############################################################################

if __name__ == "__main__":
    ret = main()
    sys.exit(ret)

###############################################################################

# Old code from before

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 4000, INPUT_FEATURES, 30, 1

# Create random Tensors to hold inputs and outputs, and wrap them in Variables.
# x = Variable(torch.randn(N, D_in))
# y = Variable(torch.randn(N, D_out), requires_grad=False)

# Create numpy arrays that hold normalized inputs and outputs, and wrap them in
# Variables.
x = Variable(torch.FloatTensor(X[:N]) / 255.0)
y = np.array(Y[:N], dtype=np.float32)
y.reshape((-1,1))
y = y / INPUT_FEATURES
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
y_pred_denormalized = torch.round(y_pred.data * float(INPUT_FEATURES)).numpy()
y_denormalized = torch.round(y.data * float(INPUT_FEATURES)).view((-1,1)).numpy()
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
y = y / INPUT_FEATURES
y = Variable(torch.from_numpy(y), requires_grad=False)

y_pred = model(x)

y_pred_denormalized = torch.round(y_pred.data * float(INPUT_FEATURES)).numpy()
y_denormalized = torch.round(y.data * float(INPUT_FEATURES)).view((-1,1)).numpy()
y_correct = (y_pred_denormalized == y_denormalized)
print("shape of y_pred:", y_pred_denormalized.shape)
print("shape of y:", y_denormalized.shape)
print("shape of y_correct:", y_correct.shape)
print(np.hstack((y_pred_denormalized, y_denormalized, y_correct)))
test_correct = int(np.sum(y_correct))
test_total = y_correct.shape[0]
print("%d correct out of %d (%2f %%)" %
      (test_correct, test_total, float(test_correct)/test_total*100.0))



# Old decision tree classifier stuff

# >>> from sklearn import tree
# >>> X = [[0, 0], [1, 1]]
# >>> Y = [0, 1]
# >>> clf = tree.DecisionTreeClassifier()
# >>> clf = clf.fit(X, Y)

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
