import matplotlib as mpl
mpl.use('Agg')
import os
import time
import glob
import tensorflow as tf
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

figsize = (6, 4)
data_dir = os.path.join('data')
figure_dir = os.path.join('figures')

def get_section_results(file, *tags):
    """
        requires tensorflow==1.12.0
    """
    data_dict = {tag: [] for tag in tags}
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag in data_dict:
                data_dict[v.tag].append(v.simple_value)
    data_dict = {tag: np.array(data_dict[tag]) for tag in data_dict}
    return data_dict

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch')
    args = parser.parse_args()
    args = vars(args)
    batch = args['batch']
    # Read data
    full_data = pd.DataFrame()

    for folder in os.listdir(data_dir):
        split = [s.strip() for s in folder.split('_')]
        if 'MsPacman-v0' in split and batch in split:
            config_list = split[split.index('q1'):split.index('MsPacman-v0')+1]
            config = '_'.join(config_list)

            logdir = os.path.join(data_dir, folder, 'events*')
            print(logdir)
            eventfile = glob.glob(logdir)[0]
            data_dict = get_section_results(eventfile,
                'Train_EnvstepsSoFar', 'Train_AverageReturn', 'Train_BestReturn')
            idx = 'Train_EnvstepsSoFar'
            for key in data_dict:
                while len(data_dict[key]) < len(data_dict[idx]):
                    data_dict[key] = np.insert(data_dict[key], 0, None)

            data = pd.DataFrame({'Iteration': range(len(data_dict[idx])),
                                 'Config': np.repeat(config, len(data_dict[idx])),
                                 **data_dict})
            full_data = pd.concat([full_data, data], axis=0, ignore_index=True)

    q1_longform = full_data.drop('Config', axis=1).melt(
        id_vars=['Train_EnvstepsSoFar'],
        value_vars=['Train_AverageReturn', 'Train_BestReturn'])

    plt.figure(figsize=figsize)
    ax = sns.lineplot(data=q1_longform, x='Train_EnvstepsSoFar', y='value', hue='variable')
    ax.set(xlabel='Training Steps', ylabel='Reward')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.savefig(os.path.join(figure_dir, 'q1.pdf'), bbox_inches='tight')

if __name__ == "__main__":
    main()