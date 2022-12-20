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

    sb.set()
    sb.set_theme(context='paper', style='whitegrid')

    if prefix is not None:
        prefix = prefix+'_'
    else:
        prefix = ''

    plt.rc('font', family='serif', serif='Times')
    plt.rc('text', usetex=True)
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('axes', labelsize=8)

    sb.set(font_scale=1.25)

    df_list = []

    for f in files:
        df = pd.read_csv(f)
        df_list.append(df)

    df = pd.concat(df_list, ignore_index=True, sort=False)

    df['send-rate-ms'] = (df['send-rate'] * 1000).astype(int)

    # plot vs cost exponent
    n = len(set(df['method']))
    drop_rates = set(df['drop-rate'])
    # colour_list = distinctipy.get_colors(n)  # colours.values(n)
    colour_list = ['#40b040', '#ba00ba', '#ff6fff']
    sb.set_style("whitegrid")
    for drop_rate in drop_rates:
        for col, ylabel, ymax in [('rmse', 'RMS Error (m)', 2), ('max-error', 'Max RMS Error (m)', None)]:
            fig, ax = plt.subplots()
            fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)

            slice = df.loc[df['drop-rate'] == drop_rate]
            # slice = slice.loc[slice['send-rate'] <= 0.5]
            sb.boxplot(x='send-rate-ms', y=col, hue='method', data=slice, palette=colour_list, linewidth=2.5)

            ax.set_xlabel("Send Rate (mS)")
            ax.set_ylabel(ylabel)
            handles, labels = ax.get_legend_handles_labels()
            for i, label in enumerate(labels):
                if label == 'fc':
                    labels[i] = 'DeadReckoning'
                if label == 'nn':
                    labels[i] = 'Ours (FC)'
                if label == 'lstm':
                    labels[i] = 'Ours (LSTM)'
            # ax.set_ylim((0.0, ymax))
            ax.legend(handles=handles, labels=labels, title='Prediction Method')
            fig.set_size_inches(width, height)
            fig.savefig('{}plot_{}_{}.pdf'.format(prefix, col, drop_rate))

    # for col, ylabel, ymax in [('rmse', 'RMS Error', 2)]:
    #     fig, ax = plt.subplots()
    #     fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)

    #     sb.boxplot(x='send-rate', y=col, hue='method', data=df, palette=colour_list, linewidth=2.5)

    #     ax.set_xlabel("Send Rate")
    #     ax.set_ylabel(ylabel)
    #     handles, labels = ax.get_legend_handles_labels()
    #     for i, label in enumerate(labels):
    #         if label == 'fc':
    #             labels[i] = 'DeadReckoning'
    #         if label == 'nn':
    #             labels[i] = '2Layer Net'
    #         if label == 'lstm':
    #             labels[i] = 'LSTM (128)'
    #     ax.set_ylim((0.0, ymax))
    #     ax.legend(handles=handles, labels=labels, title='Prediction Error -- 0-0.1% Dropped'.format(int(drop_rate*100)))
    #     fig.set_size_inches(width, height)
    #     fig.savefig('{}plot_{}_{}_all.pdf'.format(prefix, col, drop_rate))


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
