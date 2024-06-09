from typing import List, Dict, Tuple, Mapping, Sequence, Iterator, Literal

import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.model_selection import GridSearchCV
from skimage.feature import hog

import cv2

class HOG_Preprocessing(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        target_size: tuple[int, int] = (128, 128),
        orientations: int = 9,
        pixels_per_cell: tuple[int, int] = (8, 8),
        cells_per_block: tuple[int, int] = (2, 2),
        scaler_type: Literal['minmax', 'normalize'] | None = None
    ):
        self.target_size = target_size
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.scaler_type = scaler_type
        self.scaler = None

    def compute_hog_features(self, image):
        hog_features, _ = compute_hog_features(
            image = image,
            orientations = self.orientations,
            pixels_per_cell = self.pixels_per_cell,
            cells_per_block = self.cells_per_block,
            target_size = self.target_size
        )
        return hog_features

    def fit(self, X, y = None):
        if self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.scaler_type == 'normalize':
            self.scaler = Normalizer()
        elif self.scaler_type is None:
            return self
        else:
            raise ValueError("Invalid scaler.")

        X_hog = [self.compute_hog_features(image) for image in X]
        X_hog = np.array(X_hog)
        self.scaler.fit(X_hog, y)
        return self

    def transform(self, X):
        X_hog = [self.compute_hog_features(image) for image in X]
        X_hog = np.array(X_hog)
        if self.scaler is None:
            return X_hog
        return self.scaler.transform(X_hog)


class Scaler(BaseEstimator, TransformerMixin):
    def __init__(self, scaler_type = None):
        self.scaler_type = scaler_type
        self.scaler = None

    def fit(self, X, y = None):
        if self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.scaler_type == 'normalize':
            self.scaler = Normalizer()
        elif self.scaler_type is None:
            return self
        else:
            raise ValueError("Invalid scaler type.")

        self.scaler.fit(X, y)
        return self

    def transform(self, X):
        if self.scaler is None:
            return X
        return self.scaler.transform(X)


def get_image_paths(path):
    paths = []

    image_names = os.listdir(path)
    for image_name in image_names:
        image_path = os.path.join(path, image_name)
        paths.append(image_path)
    
    return paths


def show(
    items: List,
    nrows: int = 1, ncols: int = 1,
    figsize: Tuple[float, float] | None = None,
    is_random: bool = False
) -> None:
    """
    Display a grid of images in a figure window.

    Parameters
    ----------
    items : list
        A container of paths to images or a container of images matrix.

    nrows, ncols : int, default: 1
        Number of rows/columns of the subplot grid.

    figsize : (float, float), default: (nrows * 4, ncols * 4)
        Width, height in inches.

    random : bool, default: False
        Whether images should be shown randomly or not.
    """

    fig = plt.gcf()

    if figsize is None:
        fig.set_size_inches(ncols * 3, nrows * 3)
    else:
        fig.set_size_inches(figsize[0], figsize[1])

    if is_random:
        random.shuffle(items)

    for i, item in enumerate(items[: nrows * ncols]):
        plt.subplot(nrows, ncols, i + 1)

        ax = plt.gca()
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

        image = plt.imread(item) if type(item) is str else np.array(item)
        plt.imshow(image, cmap = None if image.ndim != 2 else 'gray')

    plt.show()


def plot_confusion_matrices(matrices: list, titles: list[str], labels: list[str]): 
    """
    Display many confusion matrices in a row.

    Parameters
    ----------
    matrices : Sequence of confusion matrices
        Input is a sequence of ndarrays or a sequence of nested lists.

    titles : list of strings
        Corresponding title of each confusion matrix.

    labels : list of strings
        Labels of the class.
    """

    # Create subplots and colorbar
    fig, axes = plt.subplots(1, len(matrices), figsize=(6 * len(matrices), 6))
    fig.suptitle("Confusion Matrix", fontsize=14)
    bar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.78])
    axes = np.atleast_1d(axes)

    # Normalize confusion matrices for consistent colorbar scale
    arr_t = np.concatenate(matrices)
    norm = plt.Normalize(arr_t.min(), arr_t.max())

    for i, (cm, ax) in enumerate(zip(matrices, axes.flat)):
        ax.imshow(cm, cmap=plt.cm.Blues, norm=norm)
        ax.set_title(titles[i])
        ax.set_xlabel("Predicted Label")
        ax.set_xticks(np.arange(len(labels)), labels, rotation=45)
        ax.set_yticks([])

        # Annotate
        thresh = arr_t.max() / 2
        for r in range(len(cm)):
            for c in range(len(cm[0])):
                ax.text(c, r, cm[r, c],
                         ha='center', va='center',
                         color="white" if cm[r, c] > thresh else "black")

    axes[0].set_ylabel("True Label")
    axes[0].set_yticks(np.arange(len(labels)), labels)

    # Create colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=norm)
    fig.colorbar(sm, cax=bar_ax, label="Count")

    plt.subplots_adjust(wspace=0.08)
    plt.show()


def compute_hog_features(
    image,
    orientations: int = 9,
    pixels_per_cell: tuple[int, int] = (8, 8),
    cells_per_block: tuple[int, int] = (2, 2),
    resize: bool = True,
    target_size: tuple[int, int] = (128, 128)
) -> tuple[np.ndarray, np.ndarray]:

    image = np.array(image)
    if resize:
        image = cv2.resize(image, target_size)

    # If RGB image, then convert it into grayscale image
    if image.ndim == 3:  # RBG image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Create HOG features
    hog_features, hog_image = hog(
        image = image,
        orientations = orientations,
        pixels_per_cell = pixels_per_cell,
        cells_per_block = cells_per_block,
        visualize = True
    )

    return hog_features, hog_image


def heading_key_results(tuner: GridSearchCV, n: int = 5):
    results_df = pd.DataFrame(tuner.cv_results_)

    # Create new feature showing the difference between train score and test score
    results_df['train_test_diff'] = np.abs(results_df['mean_train_score'] - results_df['mean_test_score'])

    # Choose key features to show
    param_features = list(tuner.param_grid.keys())
    for i, param in enumerate(param_features):
        new_param = param.split('__')[-1]
        param_features[i] = new_param
        results_df[new_param] = results_df['param_' + param]

    features = param_features + ['mean_train_score', 'mean_test_score', 
                                 'train_test_diff', 'rank_test_score']
    heading_results = results_df[features].sort_values(by=['rank_test_score', 'train_test_diff'])

    return heading_results.head(n)
