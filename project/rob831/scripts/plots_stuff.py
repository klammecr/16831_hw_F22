import argparse
import glob
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    X = list(range(0,15))
    Y = []
    i = 0
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            # i+=1
            # if v.tag == 'Train_EnvstepsSoFar':
            #     X.append(i)
            if v.tag == 'Eval_AverageReturn':
                Y.append(v.simple_value)
    return X, Y

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--logdir', type=str, required=True, help='path to directory contaning tensorboard results (i.e. data/q1)')
    # args = parser.parse_args()

    eventfile = glob.glob("/home/klammerc/dev/16831_hw_F22/hw4/run_logs/hw4_q5*/*events*")

    for file in eventfile:
        # if "ensemble1" in file:
        #     label = 1
        # if "ensemble3" in file:
        #     label = 3
        # if "ensemble5" in file:
        #     label = 5
        # if "horizon5" in file:
        #     label = 5
        # if "horizon15" in file:
        #     label = 15
        # if "horizon30" in file:
        #     label = 30
        if "cem_2" in file:
            label = "CEM 2 Iterations"
        if "cem_4" in file:
            label = "CEM 4 Iterations"
        elif "random" in file:
            label = "Random Actions Sample from Uniform"

        import numpy as np
        X, Y = get_section_results(file)
        i = min(len(X), len(Y))
        plt.plot(X[:i], Y[:i], label = label)
        plt.xlabel("Iteration")
        plt.ylabel("Evaluation Average Return")
        plt.legend()
        plt.title("Effect of CEM Iterations on Average Eval Return")
        plt.savefig('CEM.png')