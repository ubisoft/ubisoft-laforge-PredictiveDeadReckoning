from enum import IntEnum
import re
import sys
import argparse
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from distinctipy import distinctipy


mpl.use('pdf')


# width as measured in inkscape
width = 8  # 3.487
height = width / 1.5


def parse_state_data(files, prefix):

    sb.set_theme(style="whitegrid")
    sb.set()

    if prefix is not None:
        prefix = prefix+'_'
    else:
        prefix = ''

    plt.rc('font', family='serif', serif='Times')
    plt.rc('text', usetex=True)
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('axes', labelsize=8)

    df_list = []

    for f in files:
        df = pd.read_csv(f)
        df_list.append(df)

    df = pd.concat(df_list, ignore_index=True, sort=False)

    # plot vs cost exponent
    n = len(set(df['method']))
    colour_list = distinctipy.get_colors(n)  # colours.values(n)
    for col, ylabel in [('err', 'RMS Error'), ('ratio', 'RMSE Ratio to DeadReckoning')]:
        fig, ax = plt.subplots()
        fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)

        sb.lineplot(x='t', y=col, hue='method', data=df, palette=colour_list, linewidth=2.5)

        ax.set_xlabel("Time")
        ax.set_ylabel("RMS Error")
        handles, labels = ax.get_legend_handles_labels()
        for i, label in enumerate(labels):
            if label == 'fc':
                labels[i] = 'DeadReckoning'
            if label == 'nn':
                labels[i] = '2Layer Net'
            if label == 'lstm':
                labels[i] = 'LSTM (128)'
        ax.legend(handles=handles, labels=labels, title='Comparative RMSE')
        fig.set_size_inches(width, height)
        fig.savefig('{}plot_rmse_{}_{}.pdf'.format(prefix, col, label))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Vehicle State Parser")
    parser.add_argument(
        '-f', '--file', nargs='*', default=[],
        help='list of stat files to load')
    parser.add_argument(
        '-p', '--prefix', default=None,
        help='prefix to add to graph names')

    args = parser.parse_args()
    parse_state_data(args.file, args.prefix)
