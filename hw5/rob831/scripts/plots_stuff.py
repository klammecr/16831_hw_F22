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

    eventfile = glob.glob("/home/klammerc/dev/16831_hw_F22/hw5/data/hw5_expl_q1*hard*/*events*")

    for file in eventfile:
        if "cql_noscaleshift" in file:
            label = "CQL No Scale Shift"
        elif "cql" in file:
            label = "CQL Scale Shift"
        else:
            label = "DQN"

        import numpy as np
        X, Y = get_section_results(file)
        i = min(len(X), len(Y))
        plt.plot(X[:i], Y[:i], label = label)
        plt.xlabel("Iteration")
        plt.ylabel("Evaluation Average Return")
        plt.legend()
        plt.title("Effect of Lambda on Average Eval Return")
        plt.savefig('Q1_Extra_Hard.png')