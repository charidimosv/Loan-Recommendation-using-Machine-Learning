import matplotlib.pyplot as plt
import pandas as pd

from src.utils.conf import *


def save_figure(filename):
    plt.savefig(PICTURE_PATH + filename + FORMAT_PNG)


def save_scatter_plot_between_old(df, x_column, y_column):
    data = pd.concat([df[x_column], df[y_column]], axis=1)
    data.plot.scatter(x=x_column, y=y_column, ylim=(0, 800000))
    save_figure(y_column + "_" + x_column)


def save_scatter_plot_between(df, x_column, y_column):
    fig, ax = plt.subplots()
    ax.scatter(x=df[x_column], y=df[y_column])
    plt.xlabel(x_column, fontsize=13)
    plt.ylabel(y_column, fontsize=13)
    save_figure(y_column + "_" + x_column)
