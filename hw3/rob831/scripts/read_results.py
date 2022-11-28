import argparse
import glob
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            elif v.tag == 'Train_AverageReturn':
                Y.append(v.simple_value)
        if len(X) > 120:
            break
    return X, Y

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, required=True, help='path to directory contaning tensorboard results (i.e. data/q1)')
    args = parser.parse_args()

    eventfile = glob.glob("/home/klammerc/dev/16831_hw_F22/hw3/data/*/events*")

    for file in eventfile:
        if "hparam1" in file:
            label = 32
        if "hparam2" in file:
            label = 64
        if "hparam3" in file:
            label = 128
        if "hparam4" in file:
            label = 256
        X, Y = get_section_results(file)
        i = min(len(X), len(Y))
        plt.plot(X[:i], Y[:i], label = label)
        plt.xlabel("Iteration")
        plt.ylabel("Average Return")
        plt.legend()
        plt.title("Hidden Layer Size Effect on Average Return for 2 Hidden Layer DQN Networks")
