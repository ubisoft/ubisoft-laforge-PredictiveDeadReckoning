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

    sb.set_theme(style="darkgrid")
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
    df = df.loc[
        (df['model'] == '26_05_2022_23_14_39.pb') +
        (df['model'] == 'good_model.pb') +
        (df['model'] == '25_05_2022_10_23_07.pb') +
        (df['model'] == '26_09_2019_17_12_38.pb') +
        (df['model'] == '2019-10-16-send_rate_0_4.pb')
    ]

    # plot vs cost exponent
    n = len(set(df['model']))
    drop_rates = set(df['drop-rate'])
    colour_list = distinctipy.get_colors(n)  # colours.values(n)
    for drop_rate in drop_rates:
        for col, ylabel in [('rmse', 'RMS Error'), ('max-error', 'Max Error')]:
            fig, ax = plt.subplots()
            fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)

            slice = df.loc[(df['drop-rate'] == drop_rate) * (df['method'] == 'nn')]
            sb.lineplot(x='send-rate', y=col, hue='model', data=slice, palette=colour_list, linewidth=2.5)

            ax.set_xlabel("Send Rate")
            ax.set_ylabel(ylabel)
            handles, labels = ax.get_legend_handles_labels()
            for i, label in enumerate(labels):

                if label == 'fc':
                    labels[i] = 'DeadReckoning'
                elif label == 'nn':
                    labels[i] = '2Layer Net'
                elif label == 'lstm':
                    labels[i] = 'LSTM (128)'
                else:
                    labels[i] = label.replace('_', '-')

            ax.set_ylim((0.0, 2))
            ax.legend(handles=handles, labels=labels, title='Prediction Error -- {}\% Dropped'.format(int(drop_rate*100)))
            fig.set_size_inches(width, height)
            fig.savefig('{}plot_{}_{}.pdf'.format(prefix, col, drop_rate))


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
