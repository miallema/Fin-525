import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter


def plot_1ecdf(df, stock_name, ax):
    '''
    Computes and plots the 1 minus ECDF of a given stock.

    Parameters
    ----------
    df: Pandas DataFrame
        DataFrame containing log-returns.
    stock_name: string
        Name of the stock to plot.
    ax: matplotlib ax
        Subplot ax to plot on.
    '''
    ecdf = ECDF(abs(df[stock_name]))
    ax.plot(ecdf.x[:-1], np.log(1-ecdf.y[:-1]))
    ax.set_xlabel('|r|')
    ax.set_ylabel('log P_>(|r|)')
    ax.set_title('1 - ECDF for {}'.format(stock_name))

def plot_losses(losses):
    '''
    Plots the epoch losses.

    Parameters
    ----------
    losses: array_like
        An array containing the losses for each epoch.
    '''
    fig, ax = plt.subplots(figsize=(10,7))
    ax.plot(np.arange(len(losses)), losses)
    ax.set_title('Losses')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')
    plt.show()

def plot_cumsums(cumsum_df, year, window_size):
    '''
    Plots cumulative sum of sign returns and their predictions over one year.

    Parameters
    ----------
    cumsum_df:
        DataFrame containing both calculated cumsums of ground truth and model predictions.
    year: int
        The year to predict.
    window_size: int
        Size of prediction window.
    '''
    stocks = ['ABBN', 'CSGN', 'NESN', 'NOVN']

    fig, ax = plt.subplots(nrows=4, figsize=(15,15))
    plt.tight_layout()
    fig.suptitle('Rolling window cumsum plot from {} to {}'.format(year, year+1), fontsize=16)

    ax[0].plot(cumsum_df['ABBN'], c='blue')
    ax[1].plot(cumsum_df['CSGN'], c='blue')
    ax[2].plot(cumsum_df['NESN'], c='blue')
    ax[3].plot(cumsum_df['NOVN'], c='blue')

    for start_idx in range(window_size-1, len(cumsum_df)-window_size, window_size):
        end_idx = start_idx + window_size
        slice_df = cumsum_df.iloc[start_idx:end_idx]
        for stock_name in stocks:
            slice_df[stock_name + '_pred'] = slice_df[stock_name + '_pred'] + cumsum_df.iloc[start_idx-1][stock_name]
        ax[0].plot(slice_df['ABBN_pred'], c='red', linestyle='--')
        ax[1].plot(slice_df['CSGN_pred'], c='red', linestyle='--')
        ax[2].plot(slice_df['NESN_pred'], c='red', linestyle='--')
        ax[3].plot(slice_df['NOVN_pred'], c='red', linestyle='--')

    ax[0].set_title('ABBN')
    ax[1].set_title('CSGN')
    ax[2].set_title('NESN')
    ax[3].set_title('NOVN')

    ax[3].xaxis.set_major_locator(mdates.WeekdayLocator())
    myFmt = DateFormatter("%d.%m.%y")
    ax[3].xaxis.set_major_formatter(myFmt)
    fig.autofmt_xdate()

    # Legend
    truth_patch = mpatches.Patch(color='blue', label='Ground truth')
    preds_patch = mpatches.Patch(color='red', label='Rolling predictions')
    fig.legend(handles=[truth_patch, preds_patch], loc='lower center', ncol=2)
    fig.subplots_adjust(bottom=0.07, top=0.94)

    plt.show()
    fig.savefig('./plots/rolling_{}.pdf'.format(year), bbox_inches='tight')
