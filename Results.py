import numpy as np
import pandas as pd
import os
import glob
import Config as cfg
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter


def load_csv_files(inference=False):
    path = cfg.FINAL_RESULTS_DIR
    if cfg.WINDOWS is True and os.sep == '\\' and '\\\\?\\' not in path:
        path = '\\\\?\\' + path

    csv_files = glob.glob(os.path.join(path, "**\\*.csv"), recursive=True)

    csv_dict = {}
    # loop over the list of csv files
    for f in csv_files:
        if inference is True:
            if 'epochs-50' in f:
                continue
        # read the csv file
        df = pd.read_csv(f).to_dict()
        key = f.split("\\")[-1]
        csv_dict[key] = df

    return csv_dict


def set_plot_attributes(ax, xticks, yticks, title, xlabel, ylabel):
    # loss
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel, labelpad=1)
    if xticks is not None:
        ax.set_xticks(xticks)
        ax.set_xlim(xticks[0], xticks[len(xticks) - 1])
    if yticks is not None:
        ax.set_yticks(yticks)
        ax.set_ylim(yticks[0], yticks[len(yticks) - 1])
    ax.grid()


def plot_results(csv_dict, gpu=0, layer=False):
    num_points = 200
    epochs = np.arange(0, num_points)
    if num_points + 1 > 120:
        xticks = np.arange(0, num_points + 10, 10)
        fig_size = (30, 30)
    elif num_points + 1 > 20:
        xticks = np.arange(0, num_points + 10, 10)
        fig_size = (20, 30)
    else:
        xticks = np.arange(0, num_points + 1, 1)
        fig_size = (15, 30)
    yticks_top1 = np.arange(0, 105, 5)
    yticks_top5 = np.arange(0, 105, 5)
    yticks_loss = np.arange(0, 5.5, 0.5)

    fig_size = (50, 50)
    fig, (axs0, axs1, axs2, axs3, axs4, axs5) = plt.subplots(6, 1, figsize=fig_size)
    fig.suptitle(f'{cfg.DIR} Experiments Convolution Results', size='x-large', weight='bold')
    fig.tight_layout(pad=8)

    if layer is False:
        # Mantissa
        x_list = [23, 0, 2, 4, 7]
    else:
        # Layer
        x_list = [0, 1, 2, 3, 4, 5]

    for key in csv_dict.keys():

        if layer is False:
            # Compute flavour
            index = int(key.split("_")[0])
        else:
            index = int(key.split("_")[-1][:-4])
        x_label = x_list[index]

        # loss train
        set_plot_attributes(axs0, xticks, yticks_loss, 'Loss Train', 'Epoch', 'Loss')
        axs0.plot(epochs, csv_dict[key]['Loss_l'].values(), marker='.', label=x_label)
        axs0.legend()

        # top1 train
        set_plot_attributes(axs1, xticks, yticks_top1, 'Accuracy Top1 Train', 'Epoch', 'Accuracy')
        axs1.yaxis.set_major_formatter(PercentFormatter())
        axs1.plot(epochs, [v*100 for v in csv_dict[key]['Top1_l'].values()], marker='.', label=x_label)
        axs1.legend()

        # top5 train
        set_plot_attributes(axs2, xticks, yticks_top5, 'Accuracy Top5 Train', 'Epoch', 'Accuracy')
        axs2.yaxis.set_major_formatter(PercentFormatter())
        axs2.plot(epochs, [v*100 for v in csv_dict[key]['Top5_l'].values()], marker='.', label=x_label)
        axs2.legend()

        # loss test
        set_plot_attributes(axs3, xticks, yticks_loss, 'Loss Test', 'Epoch', 'Loss')
        axs3.plot(epochs, csv_dict[key]['Loss_t'].values(), marker='.', label=x_label)
        axs3.legend()

        # top1 test
        set_plot_attributes(axs4, xticks, yticks_top1, 'Accuracy Top1 Test', 'Epoch', 'Accuracy')
        axs4.yaxis.set_major_formatter(PercentFormatter())
        axs4.plot(epochs, [v*100 for v in csv_dict[key]['Top1_t'].values()], marker='.', label=x_label)
        axs4.legend()

        # top5 test
        set_plot_attributes(axs5, xticks, yticks_top5, 'Accuracy Top5 Test', 'Epoch', 'Accuracy')
        axs5.yaxis.set_major_formatter(PercentFormatter())
        axs5.plot(epochs, [v*100 for v in csv_dict[key]['Top5_t'].values()], marker='.', label=x_label)
        axs5.legend()

    plt.savefig(cfg.FINAL_RESULTS_DIR + "\\" + f"{cfg.DIR}_Results.png")


def plot_best_results(csv_dict, inference=False, gpu=0, layer=False):

    if not layer:
        value = ['0', '2', '4', '7', '23']
    else:
        value = ['0', '1', '2', '3', '4', '5']

    if not inference:
        max_dict = {}
        keys = list(csv_dict.keys())
        keys = np.roll(keys, -1)

        for i, key in enumerate(keys):
            val_dict = {}
            for val in csv_dict[key]:
                if val == 'Epoch':
                    continue
                dict_vals = [v for v in csv_dict[key][val].values()]
                best_val = min(dict_vals) if val == 'Loss_t' or val == 'Loss_l' else max(dict_vals)
                # best_index = np.argmin(dict_vals) if val == 'Loss_t' or val == 'Loss_l' else np.argmax(dict_vals)
                # val_dict[val] = (best_val, csv_dict[key]['Epoch'][best_index])
                val_dict[val] = best_val
            max_dict[value[i]] = val_dict
        print(pd.DataFrame(max_dict))
        print(max_dict)

    else:
        max_dict = csv_dict

    # loss_t_dict = []
    # top1_t_dict = []
    # top5_t_dict = []
    # for flavour in max_dict.keys():
    #     if flavour == '0_CM_result.csv':
    #         continue
    #     loss_t_dict.append(max_dict[flavour]['Loss_t'][0])
    #     top1_t_dict.append(max_dict[flavour]['Top1_t'][0])
    #     top5_t_dict.append(max_dict[flavour]['Top5_t'][0])
    # loss_t_dict.append(max_dict['0_CM_result.csv']['Loss_t'][0])
    # top1_t_dict.append(max_dict['0_CM_result.csv']['Top1_t'][0])
    # top5_t_dict.append(max_dict['0_CM_result.csv']['Top5_t'][0])
    #
    # fig_size = (30, 30)
    # fig, (axs0, axs1, axs2) = plt.subplots(3, 1, figsize=fig_size)
    # x_axis = [0, 2, 4, 7, 15, 23]
    # fig.suptitle(f'{cfg.EXPERIMENT} Experiments Best Results', size='x-large', weight='bold')
    # fig.tight_layout(pad=8)
    #
    # axs0.set_title('Best Loss Convolution Results', size='x-large', weight='bold')
    # axs0.plot(loss_t_dict, marker='.', color='blue')
    # axs0.set_xticks(np.arange(len(loss_t_dict)), x_axis)
    # axs0.set_xlabel('x_label', size='x-large', weight='bold')
    # axs0.set_ylabel('Loss', size='x-large', weight='bold')
    #
    # axs1.set_title('Best Accuracy Top1 Convolution Results', size='x-large', weight='bold')
    # axs1.plot(top1_t_dict, marker='.', color='blue')
    # axs1.set_xticks(np.arange(len(top1_t_dict)), x_axis)
    # axs1.set_xlabel('x_label', size='x-large', weight='bold')
    # axs1.set_ylabel('Top1', size='x-large', weight='bold')
    #
    # axs2.set_title('Best Accuracy Top5 Convolution Results', size='x-large', weight='bold')
    # axs2.plot(top5_t_dict, marker='.', color='blue')
    # axs2.set_xticks(np.arange(len(top5_t_dict)), x_axis)
    # axs2.set_xlabel('x_label', size='x-large', weight='bold')
    # axs2.set_ylabel('Top5', size='x-large', weight='bold')
    #
    # plt.savefig(cfg.FINAL_RESULTS_DIR + "\\" + f"{cfg.EXPERIMENT}_Best_Results.png")

    # save best results to csv file
    pd.DataFrame(max_dict).to_csv(cfg.FINAL_RESULTS_DIR + "\\" + f"{cfg.DIR}_Best_Results.csv")


def main():
    csv_dict = load_csv_files(inference=False)
    # plot_results(csv_dict)
    # plot_best_results(csv_dict, inference=False)
    plot_results(csv_dict, layer=True)
    plot_best_results(csv_dict, inference=False, layer=True)


if __name__ == "__main__":
    main()
