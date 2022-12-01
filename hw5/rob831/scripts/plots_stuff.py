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
    i = 0
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            # i+=1
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            if v.tag == 'Eval_AverageReturn':
                Y.append(v.simple_value)
    return X, Y

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--logdir', type=str, required=True, help='path to directory contaning tensorboard results (i.e. data/q1)')
    # args = parser.parse_args()

    eventfile = glob.glob("/home/klammerc/dev/16831_hw_F22/hw5/data/hw5_expl_q4*medium*/*events*")

    for file in eventfile:
        if "lam0.1" in file:
            label = "0.1"
        if "lam1" in file:
            label = "1"
        if "lam2" in file:
            label = "2"
        if "lam10" in file:
            label = "10"
        if "lam20" in file:
            label = "20"
        if "lam50" in file:
            label = "50"

        import numpy as np
        X, Y = get_section_results(file)
        i = min(len(X), len(Y))
        plt.plot(X[:i], Y[:i], label = label)
        plt.xlabel("Iteration")
        plt.ylabel("Evaluation Average Return")
        plt.legend()
        plt.title("Effect of Lambda on Average Eval Return")
        plt.savefig('Q4_Medium.png')