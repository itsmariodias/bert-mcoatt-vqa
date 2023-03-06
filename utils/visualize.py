from matplotlib import pyplot as plt


# Loss Graphs
def plot_metric_graph(log, savefile, metric='loss', show_val=True):
    """
    Plot graph for the given metric from log file.
    """
    plt.figure()
    plt.plot(log[metric])
    if show_val: plt.plot(log['val_'+metric])
    plt.title('model '+metric)
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(savefile)
