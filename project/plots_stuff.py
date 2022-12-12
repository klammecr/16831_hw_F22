import argparse
import glob
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.summary.summary_iterator import summary_iterator

def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    i = 0
    for e in summary_iterator(file):
        X.append(i)
        i+=1
        for v in e.summary.value:
               
            # if v.tag == 'Train_EnvstepsSoFar':
            #     X.append(v.simple_value)
            if v.tag == 'Eval_AverageReturn':
                Y.append(v.simple_value)
    return X, Y

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--logdir', type=str, required=True, help='path to directory contaning tensorboard results (i.e. data/q1)')
    # args = parser.parse_args()

    # eventfile = glob.glob("/home/klammerc/dev/16831_hw_F22/project/data/resultsMB_Exp/*events*")

    # for file in eventfile:

    import numpy as np
    X, Y = get_section_results("/home/klammerc/dev/16831_hw_F22/project/data/resultsMB_Exp_navigation-aviary-v0_12-12-2022_02-40-50/events.out.tfevents.1670830850.klammerc-B450-AORUS-PRO-WIFI")
    i = min(len(X), len(Y))
    plt.plot(X[:i], Y[:i])
    plt.xlabel("Iteration")
    plt.ylabel("Evaluation Average Return")
    plt.legend()
    plt.title("Model-Based RL with CEM Sampling - UAV")
    plt.savefig('MB_CEM_Eval.png')