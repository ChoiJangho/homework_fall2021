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
    # Read data
    dqn_data = pd.DataFrame()
    for folder in os.listdir(data_dir):
        split = [s.strip() for s in folder.split('_')]
        # print(split)
        if 'LunarLander-v3' in split and 'dqn' in split and 'q2' in split:
            config_list = split[split.index('q2'):split.index('LunarLander-v3')+1]
            config = '_'.join(config_list)

            logdir = os.path.join(data_dir, folder, 'events*')
            print(logdir)
            eventfile = glob.glob(logdir)[0]
            data_dict = get_section_results(eventfile,
                'Train_EnvstepsSoFar', 'Train_AverageReturn')
            idx = 'Train_EnvstepsSoFar'
            for key in data_dict:
                while len(data_dict[key]) < len(data_dict[idx]):
                    data_dict[key] = np.insert(data_dict[key], 0, None)

            data = pd.DataFrame({'Iteration': range(len(data_dict[idx])),
                                 'Config': np.repeat(config, len(data_dict[idx])),
                                 **data_dict})
            dqn_data = pd.concat([dqn_data, data], axis=0, ignore_index=True)

    # Read data
    pararms1_data = pd.DataFrame()
    for folder in os.listdir(data_dir):
        split = [s.strip() for s in folder.split('_')]
        # print(split)
        if 'LunarLander-v3' in split and 'q3' in split and 'hparams11' in split:
            config_list = split[split.index('q3'):split.index('LunarLander-v3')+1]
            config = '_'.join(config_list)

            logdir = os.path.join(data_dir, folder, 'events*')
            print(logdir)
            eventfile = glob.glob(logdir)[0]
            data_dict = get_section_results(eventfile,
                'Train_EnvstepsSoFar', 'Train_AverageReturn')
            idx = 'Train_EnvstepsSoFar'
            for key in data_dict:
                while len(data_dict[key]) < len(data_dict[idx]):
                    data_dict[key] = np.insert(data_dict[key], 0, None)

            data = pd.DataFrame({'Iteration': range(len(data_dict[idx])),
                                 'Config': np.repeat(config, len(data_dict[idx])),
                                 **data_dict})
            pararms1_data = pd.concat([pararms1_data, data], axis=0, ignore_index=True)

    params2_data = pd.DataFrame()
    for folder in os.listdir(data_dir):
        split = [s.strip() for s in folder.split('_')]
        # print(split)
        if 'LunarLander-v3' in split and 'q3' in split and 'hparams12' in split:
            config_list = split[split.index('q3'):split.index('LunarLander-v3')+1]
            config = '_'.join(config_list)

            logdir = os.path.join(data_dir, folder, 'events*')
            print(logdir)
            eventfile = glob.glob(logdir)[0]
            data_dict = get_section_results(eventfile,
                'Train_EnvstepsSoFar', 'Train_AverageReturn')
            idx = 'Train_EnvstepsSoFar'
            for key in data_dict:
                while len(data_dict[key]) < len(data_dict[idx]):
                    data_dict[key] = np.insert(data_dict[key], 0, None)

            data = pd.DataFrame({'Iteration': range(len(data_dict[idx])),
                                 'Config': np.repeat(config, len(data_dict[idx])),
                                 **data_dict})
            params2_data = pd.concat([params2_data, data], axis=0, ignore_index=True)

    params3_data = pd.DataFrame()
    for folder in os.listdir(data_dir):
        split = [s.strip() for s in folder.split('_')]
        # print(split)
        if 'LunarLander-v3' in split and 'q3' in split and 'hparams13' in split:
            config_list = split[split.index('q3'):split.index('LunarLander-v3') + 1]
            config = '_'.join(config_list)

            logdir = os.path.join(data_dir, folder, 'events*')
            print(logdir)
            eventfile = glob.glob(logdir)[0]
            data_dict = get_section_results(eventfile,
                                            'Train_EnvstepsSoFar', 'Train_AverageReturn')
            idx = 'Train_EnvstepsSoFar'
            for key in data_dict:
                while len(data_dict[key]) < len(data_dict[idx]):
                    data_dict[key] = np.insert(data_dict[key], 0, None)

            data = pd.DataFrame({'Iteration': range(len(data_dict[idx])),
                                 'Config': np.repeat(config, len(data_dict[idx])),
                                 **data_dict})
            params3_data = pd.concat([params3_data, data], axis=0, ignore_index=True)

    # full_data.loc[:, 'Train_AverageReturn'] /= 3
    plt.figure(figsize=figsize)
    ax = sns.lineplot(data=pararms1_data, x='Train_EnvstepsSoFar', y='Train_AverageReturn')
    sns.lineplot(data=params2_data, x='Train_EnvstepsSoFar', y='Train_AverageReturn')
    sns.lineplot(data=params3_data, x='Train_EnvstepsSoFar', y='Train_AverageReturn')
    ax = sns.lineplot(data=dqn_data, x='Train_EnvstepsSoFar', y='Train_AverageReturn')
    ax.set(xlabel='Training Steps', ylabel='Reward')

    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.savefig(os.path.join(figure_dir, 'q3_4.pdf'), bbox_inches='tight')

if __name__ == "__main__":
    main()