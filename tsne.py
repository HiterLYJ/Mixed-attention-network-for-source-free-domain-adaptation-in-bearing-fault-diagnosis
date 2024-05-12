from sklearn import datasets
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

digits = datasets.load_digits(n_class=5)
X = digits.data
y = digits.target
print(X.shape)  # 901,64
print(type(X))



def plot_embedding_2d(X, y, title=None):
    """Plot an embedding X with the class label y colored by the domain d."""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    print(x_max)
    X = (X - x_min) / (x_max - x_min)
    # Plot colors numbers
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        # plot colored number
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set3(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.show()


def plot_embedding_3d(X, y, title=None):
    # 坐标缩放到[0,1]区间
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X = (X - x_min) / (x_max - x_min)
    # 降维后的坐标为（X[i, 0], X[i, 1],X[i,2]），在该位置画出对应的digits
    fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax = Axes3D(fig)
    for i in range(X.shape[0]):
        ax.text(X[i, 0], X[i, 1], X[i, 2], str(digits.target[i]),
                color=plt.cm.Set3(y[i] / 10.),
                fontdict={'weight': 'bold', 'size': 9})
    if title is not None:
        plt.title(title)
    plt.show()


print("Computing t-SNE embedding")
tsne2d = TSNE(n_components=2, init='pca', random_state=0)

X_tsne_2d = tsne2d.fit_transform(X)
plot_embedding_2d(X_tsne_2d[:, 0:2], y, "t-SNE 2D")
