#!/usr/bin/env python

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import sys, os, time, argparse, random
from collections import OrderedDict
import pandas as pd
import re
import bz2
import gzip

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

Input file example (modbus bytes enclosed in square braces, 1 datagram/line):
[00][01][00][00][00][08][FF][0F][00][2E][00][06][01][20]
[00][02][00][00][00][0C][FF][0F][00][39][00][23][05][49][8A][A5][71][06]
...

By default, this data is read from stdin. Alternatively, -i can be used to read
from a file instead, where the file extension may be .gz or .bz2, indicating
gzip or bzip2 compression, respectively. The input format also need not use
square brackets such as "[FF]"; the parser simply expects a sequence of
hex-encoded bytes separated by non-alphanumeric delimiters.
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
Example: 500 30 20 1""")
    parser.add_argument('-b', '--batchsize', metavar='B', type=int,
                        default=500, help="batch size (default 500)")
    parser.add_argument('-e', '--epochs', metavar='E', type=int,
                        default=200, help="number of epochs (default 200)")
    parser.add_argument('-t', '--trainsize', metavar='M', type=int,
                        default=8000,
                        help="training set size (#examples, default 8000)")
    parser.add_argument('-d', '--devsize', metavar='M', type=int,
                        default=None,
                        help="dev set size (#examples, default: the rest)")
    parser.add_argument('-l', '--learnrate', metavar='L', type=float,
                        default=1e-5, help="learning rate (default 1e-5)")
    parser.add_argument('-m', '--momentum', metavar='M', type=float,
                        default=0.9, help="Nesterov momentum (default 0.9)")
    parser.add_argument('-i', '--input', metavar='F', type=str,
                        default=None,
                        help="input file rather than stdin (handles *.gz/.bz2)")
    parser.add_argument('-o', '--outdir', metavar='D', type=str,
                        default=None,
                        help="output directory for stats and hyperparameters")
    parser.add_argument('-I', '--expID', metavar='n', type=int, default=0,
                        help="experiment ID (integer, default=0)")
    parser.add_argument('-s', '--extrastats', action='store_true',
                        help="record extra statistics (accuracy every batch)")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="be more chatty on stderr")
    args = parser.parse_args()

    INPUT_FEATURES = args.layersizes[0] # input dimension
    if args.layersizes[-1] != 1:
        eprint("Number of units per layer: ", args.layersizes)
        parser.error("The final layer must be a single unit, but got %d." %
                     args.layersizes[-1])
    if args.batchsize > args.trainsize or \
       args.trainsize % args.batchsize != 0:
        parser.error("Batch size (%d) must divide training data size (%d)" %
                     (args.batchsize, args.trainsize))
    # Print args
    if args.verbose:
        eprint("Running NN training with the following parameters:")
        eprint("-"*78)
        eprint("args =", args)
        eprint("-"*78)

    # Read input
    if args.input is not None:
        if args.input.endswith(".gz"):
            inputfp = gzip.open(args.input, mode='rt')
        elif args.input.endswith(".bz2"):
            inputfp = bz2.open(args.input, mode='rt')
        else:
            inputfp = open(args.input, "r")
    else:
        inputfp = sys.stdin
    X, Y = read_packet_lines(inputfp, INPUT_FEATURES)

    # Split into training and dev sets
    if args.devsize is None:
        args.devsize = len(X) - args.trainsize
        if args.trainsize < 0:
            eprint("Error: not enough data for %d training examples" %
                   args.trainsize, "(plus some dev examples)")
            return -1
    elif args.trainsize + args.devsize > len(X):
        eprint("Error: not enough data for %d training and %d dev examples" %
               (args.trainsize, args.devsize))
        return -1

    trainX = X[:args.trainsize]
    trainY = Y[:args.trainsize]
    devX = X[args.trainsize:(args.trainsize + args.devsize)]
    devY = Y[args.trainsize:(args.trainsize + args.devsize)]
    if args.verbose:
        eprint("Data split into %d training and %d dev examples" %
               (len(trainX), len(devX)))

    # Train a model
    model = nn_train(args, trainX, trainY, devX, devY)

    # Run the model on the training set
    trainY_predict = run_model(model, trainX)
    print("-"*78)
    print("Test-on-Train Accuracy: ", end="")
    trainAccuracy = compute_accuracy(trainY, trainY_predict, trainX,
                                     screen_output=True)

    # Run the model on the dev set
    devY_predict = run_model(model, devX)
    print("-"*78)
    print("Hold-out Cross Validation Accuracy: ", end="")
    devAccuracy = compute_accuracy(devY, devY_predict, devX,
                                   screen_output=True)

    # Write output to directory if requested
    if args.outdir is not None:
        write_output_dir(args.outdir, model, trainAccuracy, devAccuracy)

    return 0

def make_rand_bytes(n):
    # the following is faster than os.urandom(n) and can be manually seeded
    return random.getrandbits(8*n).to_bytes(n, sys.byteorder)

def pad_or_truncate(some_list, target_len, rand_pad=True):
    if len(some_list) >= target_len:
        return some_list[:target_len]
    else:
        pad_len = target_len - len(some_list)
        new_list = some_list[:]
        if rand_pad:
            new_list.extend(make_rand_bytes(pad_len))
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
        # ignore non-alphanumeric characters like "[" and "]"
        fields = re.sub(r'\W', " ", line).split()
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

def nn_train(params, X, Y, devX=None, devY=None):
    """PyTorch: nn

    A fully-connected ReLU network with several hidden layers, trained to
    predict y from x by minimizing squared Euclidean distance.

    This implementation uses the nn package from PyTorch to build the network.
    PyTorch autograd makes it easy to define computational graphs and take
    gradients, but raw autograd can be a bit too low-level for defining complex
    neural networks; this is where the nn package can help. The nn package
    defines a set of Modules, which you can think of as a neural network layer
    that has produces output from input and may have some trainable weights.

    X = training inputs (n x #features)
    Y = training outputs (n-long vector)
    devX = development / hold-out-cross-validation X (n' x #features)
    devY = development / hold-out-cross-validation Y (n'-long vector)
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
            torch.nn.init.xavier_uniform_(m.weight)
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


    # Prepare tensor versions of train and dev sets
    trainX = torch.FloatTensor(X)
    trainY = torch.FloatTensor(Y).unsqueeze(1)
    if devX is not None and devY is not None:
        devX = torch.FloatTensor(devX)
        devY = torch.FloatTensor(devY).unsqueeze(1)

    # Set optimizer
    optimizer = torch.optim.SGD(nn_model.parameters(),
                                lr = params.learnrate,
                                momentum=params.momentum, # Nesterov momentum
                                #weight_decay=1e-2 # L2 regularization
                                )

    # The nn package also contains definitions of popular loss functions; in
    # this case we will use Mean Squared Error (MSE) as our loss function.
    loss_fn = torch.nn.MSELoss(reduction='sum')

    # Get ready to collect statistics:
    # [[epoch, batch, loss, trainAccuracy, devAccuracy],
    #  [epoch, batch, loss, trainAccuracy, devAccuracy],
    #  ...
    #  [epoch, batch, loss, trainAccuracy, devAccuracy]]
    # Note that epoch and batch are 1-indexed when we record statistics
    stats = []

    # Gather parts of model to be trained
    model = {}
    model["nn_model"] = nn_model
    model["Xmean"] = Xmean
    model["Xstd"] = Xstd
    model["Ymean"] = Ymean
    model["Ystd"] = Ystd
    model["params"] = params

    ts_begin = pd.Timestamp.now()

    # Train the neural network
    for epoch in range(params.epochs):
        for batch, (Xbatch, Ybatch) in enumerate(training_loader):
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
            last_loss = loss.data.item()

            # Backward pass: compute gradient of the loss with respect to all
            # the learnable parameters of the model. Internally, the parameters
            # of each Module are stored in Variables with requires_grad=True, so
            # this call will compute gradients for all learnable parameters in
            # the model.
            loss.backward()

            # Update the weights using gradient descent.
            optimizer.step()

            # Collect statistics on loss
            elapsed_time = (pd.Timestamp.now() - ts_begin).total_seconds()
            stats_line = [elapsed_time,
                          epoch + 1,
                          batch + 1,
                          last_loss]
            batches_per_epoch = params.trainsize/params.batchsize
            last_batch = batches_per_epoch - 1
            if params.extrastats or (batch == last_batch):
                # optionally, test-on-train accuracy
                trainYpred = run_model(model, trainX)
                train_accuracy = compute_accuracy(trainY, trainYpred)
                correct = train_accuracy["numcorrect"]
                total = train_accuracy["numtotal"]
                stats_line.append(correct/total)
                if devX is not None and devY is not None:
                    # optionally, dev accuracy (hold-out cross-validation)
                    devYpred = run_model(model, devX)
                    dev_accuracy = compute_accuracy(devY, devYpred)
                    correct = dev_accuracy["numcorrect"]
                    total = dev_accuracy["numtotal"]
                    stats_line.append(correct/total)
                # Display
                eprint("Epoch=%4d" % (epoch+1),
                       "T(s)=%5.2f" % elapsed_time,
                       "Batch|%d|%2d/%d" % \
                          (params.batchsize, batch+1, batches_per_epoch),
                       " MSE= %.3e" % last_loss, end="")
                if len(stats_line) >= 5:
                    eprint(" trainAcc= %.2f%%" % (stats_line[4]*100.0), end="")
                if len(stats_line) >= 6:
                    eprint(" devAcc= %.2f%%" % (stats_line[5]*100.0), end="")
                eprint("") # newline
                # add stats line to the record
                stats.append(stats_line)

    # Convert training stats into dataframe and return it with the model.
    # Missing data (i.e., train/dev accuracy for intermediate batches) is
    # represented by a NaN in the data frame.
    col_headers = pd.Series(["Timestamp",
                             "Epoch",
                             "Batch",
                             "MSE",
                             "trainAcc",
                             "devAcc"])
    statsdf = pd.DataFrame(stats, columns=col_headers)
    model["stats"] = statsdf

    return model

def write_output_dir(outdir, model, trainAcc, devAcc):
    """Write training statistics, hyperparameters, and predictions to separate CSV
    files in the designated output directory.
    """

    # Create directory if it doesn't yet exist. Then prepare filenames.
    os.makedirs(outdir, exist_ok=True)
    training_filename = outdir + os.sep + "training_stats.csv"
    hyperparams_filename = outdir + os.sep + "hyperparams.csv"
    predictions_filename = outdir + os.sep + "predictions.csv"

    # Get experiment ID (add as the first column to each of the data frames)
    expID = model["params"].expID

    # Prepare and write training stats
    statsdf = model["stats"].copy()
    statsdf.insert(0, "expID", expID)
    append_df_to_csv(statsdf, training_filename)
    if args.verbose:
        eprint("Wrote %d stats lines to %s" % (len(statsdf), training_filename))

    # Prepare and write hyperparameters
    paramsdf = organize_hyperparams(model["params"])
    append_df_to_csv(paramsdf, hyperparams_filename)
    if args.verbose:
        eprint("Wrote %d lines to %s" % (len(paramsdf), hyperparams_filename))

    # Prepare and write predictions
    traindf = organize_accuracy(trainAcc, "train")
    devdf = organize_accuracy(devAcc, "dev")
    bothdf = pd.concat([traindf, devdf], ignore_index=True)
    dataID = pd.Series(range(len(traindf) + len(devdf)), dtype="int64")
    bothdf.insert(0, "dataID", dataID)
    bothdf.insert(0, "expID", expID)
    append_df_to_csv(bothdf, predictions_filename)
    if args.verbose:
        eprint("Wrote %d lines to %s" % (len(bothdf), predictions_filename))

    return

def append_df_to_csv(df, filename):
    # If creating a new file, need a column header line.
    need_header = not os.path.isfile(filename)
    with open(filename, 'a') as f:
        df.to_csv(f, header=need_header, index=False)

def organize_hyperparams(params):
    """Return dataframe with parameters organized"""

    # Split layer sizes into separate columns.
    MAX_LAYERS = 4
    assert len(params.layersizes) <= MAX_LAYERS
    paramdict = params.__dict__.copy()
    L = paramdict.pop("layersizes")
    if len(L) < MAX_LAYERS:
        L += [np.nan] * (MAX_LAYERS - len(L))
    for i in range(MAX_LAYERS):
        col_name = "layer" + str(i)
        paramdict[col_name] = L[i]

    # Return a data frame with the columns sorted alphabetically.
    paramdf = pd.DataFrame.from_dict({k:[v] for k,v in paramdict.items()})
    paramdf = paramdf.reindex(columns=sorted(paramdf.columns))
    return paramdf

def organize_accuracy(acc, datasetlabel):
    assert datasetlabel in ("train", "dev")
    col_labels = [datasetlabel, "Ypred", "Y", "Ycorrect"]
    accdf = pd.DataFrame({"dataset": datasetlabel,
                          "Ypred": acc["Ypred"].flatten(),
                          "Y": acc["Y"].flatten(),
                          "Ycorrect": acc["Ycorrect"].flatten()})

    from pandas.api.types import CategoricalDtype
    cat_type = CategoricalDtype(categories=["train", "dev"], ordered=True)
    accdf["dataset"] = accdf["dataset"].astype(cat_type)
    return accdf

def run_model(model, X):
    X = torch.FloatTensor(X)
    Xnorm = normalize_tensor(X, model["Xmean"], model["Xstd"])[0]
    xnorm = Variable(Xnorm, requires_grad=False)
    ynorm_pred = model["nn_model"](xnorm)
    Y_predict = denormalize_tensor(ynorm_pred.data,
                                   model["Ymean"],
                                   model["Ystd"])
    return Y_predict

def compute_accuracy(Y, Ypred, X=None, screen_output=False):
    if screen_output or args.verbose:
        sys.stderr.flush()
        sys.stdout.flush()
    Y = np.reshape(np.asarray(Y, dtype=np.int32), (-1, 1))
    Ypred = torch.round(Ypred).numpy().astype(np.int32)
    Ycorrect = (Y == Ypred).flatten()
    Yincorrect = ~Ycorrect
    num_incorrect = int(np.sum(Yincorrect))

    # compute accuracy
    test_correct = int(np.sum(Ycorrect))
    test_total = len(Ycorrect)
    if screen_output:
        print("%d correct out of %d (%2f %%)" %
              (test_correct, test_total, float(test_correct)/test_total*100.0))

    # display mistakes
    NUM_MISTAKES_DSP = 10 # number to display
    NUM_X_FEATURES = 12 # number to display
    if screen_output and args.verbose and num_incorrect > 0:
        if X is not None:
            X = np.asarray(X, dtype=np.int32)
            x_column_labels = ", X[0],...,X[%d]" % (NUM_X_FEATURES - 1)
            result_stack = np.hstack((Ypred, Y, X[:,:NUM_X_FEATURES]))
            mistake_stack = result_stack[Yincorrect][:NUM_MISTAKES_DSP]
        else:
            x_column_labels = ""
            result_stack = np.hstack((Ypred, Y))
            mistake_stack = result_stack[Yincorrect][:NUM_MISTAKES_DSP]

        eprint("First %d mistakes: [predicted, actual%s]" %
               (mistake_stack.shape[0], x_column_labels))
        eprint(mistake_stack)
        sys.stderr.flush()

    # Collect results
    results = {}
    results["numcorrect"] = test_correct
    results["numtotal"] = test_total
    results["Y"] = Y
    results["Ypred"] = Ypred
    results["Ycorrect"] = Ycorrect

    return results


###############################################################################

if __name__ == "__main__":
    ret = main()
    sys.exit(ret)

###############################################################################

# Old decision tree classifier stuff

# from sklearn import tree
# from sklearn.model_selection import cross_val_score
# from sklearn.ensemble import AdaBoostClassifier

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
