import matplotlib.pyplot as plt


# Code from https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
def plot_images(images, cls_true, cls_pred=None, weight=None):
    """
    Adapted from https://github.com/Hvass-Labs/TensorFlow-Tutorials/
    """

    fig, axes = plt.subplots(2, 5, figsize=(10,6))

    for i, ax in enumerate(axes.flat):
        # plot img
        if len(images.shape) == 3:
            ax.imshow(images[i, :, :], interpolation='spline16')
        else:
            ax.imshow(images[i, :, :, :], interpolation='spline16')

        # show true & predicted classes
        if cls_pred is None:
            xlabel = "{0}".format(cls_true[i])
        else:
            xlabel = "True: {} \nPred: {}  \nWeight: {:.4f}".format(
                cls_true[i], cls_pred[i], weight[i]
            )
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.subplots_adjust(wspace=0.2, hspace=0.2)

    plt.show()
