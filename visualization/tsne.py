import matplotlib.pyplot as plt
from sklearn import manifold, datasets



def visualization(X, y, figname, len_=-1, show=False):
    # np.random(100)

    tsne = manifold.TSNE(n_components=3, init='pca', random_state=100, learning_rate='auto', method='barnes_hut',
                         early_exaggeration=2)
    X_tsne = tsne.fit_transform(X)
    # print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)

    ax = plt.subplot(111, projection='3d')

    for i in range(X_norm.shape[0]):
        # Customize colors
        co = plt.cm.Set1(y[i])
        ax.scatter(X_norm[i, 0], X_norm[i, 1], X_norm[i, 2], color=co, marker='o')

    ax.set_zlabel('Z')
    ax.set_ylabel('Y')
    ax.set_xlabel('X')

    ax.set_facecolor((1.0, 1.0, 1.0, 0.3))

    ax.xaxis._axinfo["grid"]['color'] = (0.5, 0.5, 0.5, 0.3)
    ax.yaxis._axinfo["grid"]['color'] = (0.5, 0.5, 0.5, 0.3)
    ax.zaxis._axinfo["grid"]['color'] = (0.5, 0.5, 0.5, 0.3)

    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.3))
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.3))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.3))

    plt.savefig(fname='{}.png'.format(figname), format='png')
    plt.show()
    print('-' * 50)
