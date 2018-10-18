#!/usr/bin/env python

import random
import math
from functools import reduce
import sys
import argparse
import subprocess

description_text = \
"""\
Run a hyperparameter random search with the specified number of random points.
"""

# hyperparameter ranges and distribution
hpranges = {}
hpranges["epochs"] = {"type":"log", "round":"int",
                      "min": 300, "max": 3000}
hpranges["batchsize"] = {"type":"log", "round":"factor 2000",
                         "min": 100, "max": 2000}
hpranges["trainsize"] = {"type":"linear", "round":"multiple 2000",
                         "min": 2000, "max": 90000}
hpranges["devsize"] = {"type":"set", "set":[10000]}
hpranges["learnrate"] = {"type":"log",
                         "min": 1e-6, "max":1e-3}
hpranges["momentum"] = {"type":"1-log",
                        "min": 0, "max": 0.99}
hpranges["hiddenLayers"] = {"type":"set", "set":[1,2]}
hpranges["unitsPerLayer"] = {"type":"log", "round":"int",
                             "min": 2, "max": 50}
hpranges["firstLayer"] = {"type":"set", "set":[50, 500]}
hpranges["lastLayer"] = {"type":"set", "set":[1]}

def sampleParamDistribution(paramrange):

    # sample from distribution
    if paramrange["type"] == "linear":
        a = paramrange["min"]
        b = paramrange["max"]
        x = random.uniform(a, b)
    elif paramrange["type"] == "log":
        a = paramrange["min"]
        b = paramrange["max"]
        x = loguniform(a, b)
    elif paramrange["type"] == "1-log":
        a = 1.0 - paramrange["max"]
        b = 1.0 - paramrange["min"]
        x = 1.0 - loguniform(a, b)
    elif paramrange["type"] == "set":
        x = random.choice(paramrange["set"])

    # apply rounding
    if "round" not in paramrange:
        pass
    elif paramrange["round"] == "int":
        x = round(x)
    elif paramrange["round"].startswith("factor"):
        n = int(paramrange["round"].split()[1])
        x = round_to_nearest_factor(x, n)
    elif paramrange["round"].startswith("multiple"):
        base = int(paramrange["round"].split()[1])
        x = round_to_nearest_multiple(x, base)

    return x

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def loguniform(a, b):
    logA = math.log(a)
    logB = math.log(b)
    sample = random.uniform(logA, logB)
    return math.exp(sample)

def round_to_nearest_multiple(x, base):
    return int(base * round(float(x)/base))

def round_to_nearest_factor(x, n):
    return take_closest(x, factors(n))

def take_closest(x, L):
    return min(L, key=lambda y:abs(y-x))

def factors(n):
    """Return list of factors of n in sorted order."""
    factor_pairs = ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)
    unique_factors = set(reduce(list.__add__, factor_pairs))
    return sorted(list(unique_factors))

def sample_hyperparameters(ranges):
    special_hyperparams = set(["firstLayer", "lastLayer",
                                "hiddenLayers", "unitsPerLayer"])
    for hp in special_hyperparams:
        assert hp in ranges

    hpset = {}

    # Special hyperparameters
    layersizes = []
    layersizes.append(sampleParamDistribution(ranges["firstLayer"]))
    num_hidden_layers = sampleParamDistribution(ranges["hiddenLayers"])
    for i in range(num_hidden_layers):
        layersizes.append(sampleParamDistribution(ranges["unitsPerLayer"]))
    layersizes.append(sampleParamDistribution(ranges["lastLayer"]))
    hpset["layersizes"] = layersizes

    # Normal hyperparameters
    for hp in ranges:
        if hp in special_hyperparams:
            continue
        hpset[hp] = sampleParamDistribution(ranges[hp])

    return hpset

def run_training(hyperparams, inputfile, outdir, expID=0, verbose=False):
    cmdline = ["./learn-length.py"]
    if verbose:
        cmdline.append("-v")
    cmdline.extend(["-I", str(expID)])
    cmdline.extend(["-b", str(hyperparams["batchsize"])])
    cmdline.extend(["-e", str(hyperparams["epochs"])])
    cmdline.extend(["-t", str(hyperparams["trainsize"])])
    cmdline.extend(["-d", str(hyperparams["devsize"])])
    cmdline.extend(["-l", str(hyperparams["learnrate"])])
    cmdline.extend(["-m", str(hyperparams["momentum"])])
    cmdline.extend(["-i", inputfile])
    cmdline.extend(["-o", outdir])
    cmdline.extend([str(x) for x in hyperparams["layersizes"]])
    if verbose:
        eprint("command:", " ".join(cmdline))
    subprocess.run(cmdline)

def main():
    global args
    parser = argparse.ArgumentParser(description=description_text)
    parser.add_argument('numpoints', metavar='N', type=int,
                        help="number of points to try in hyperparameter space")
    parser.add_argument('input', type=str,
                        help="input file")
    parser.add_argument('outdir', type=str,
                        help="output directory for stats and hyperparameters")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="be more chatty on stderr")
    args = parser.parse_args()

    for i in range(args.numpoints):
        hpset = sample_hyperparameters(hpranges)
        if args.verbose:
            eprint("HyperParams %d:" % i, hpset)
        run_training(hpset, args.input, args.outdir, expID=i,
                     verbose=args.verbose)

    return 0

if __name__ == "__main__":
    sys.exit(main())
