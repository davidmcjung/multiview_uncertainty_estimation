import numbers
from sklearn.utils import check_random_state
from sklearn.utils import shuffle as util_shuffle
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def make_moons(n_training_samples=100, radius=1., shuffle=True, noise=None, random_state=None):
    """Make two interleaving half circles.
    A simple toy dataset to visualize clustering and classification
    algorithms. Read more in the :ref:`User Guide <sample_generators>`.
    Parameters. Codes are adopted from https://github.com/scikit-learn/scikit-learn/blob/80598905e/sklearn/datasets/_samples_generator.py#L724.
    ----------
    n_training_samples : int or tuple of shape (2,), dtype=int, default=100
        If int, the total number of points generated.
        If two-element tuple, number of points in each of two moons.
        .. versionchanged:: 0.23
           Added two-element tuple.
    radius : float, default=1.0
        Raidus of circle.
    shuffle : bool, default=True
        Whether to shuffle the samples.
    noise : float, default=None
        Standard deviation of Gaussian noise added to the data.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset shuffling and noise.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
    Returns
    -------
    train_X : ndarray of shape (n_training_samples, 2)
        The generated samples.
    y : ndarray of shape (n_training_samples,)
        The integer labels (0 or 1) for class membership of each sample.
    """

    if isinstance(n_training_samples, numbers.Integral):
        n_training_samples_out = n_training_samples // 2
        n_training_samples_in = n_training_samples - n_training_samples_out
    else:
        try:
            n_training_samples_out, n_training_samples_in = n_training_samples
        except ValueError as e:
            raise ValueError(
                "`n_training_samples` can be either an int or a two-element tuple."
            ) from e

    generator = check_random_state(random_state)

    outer_circ_x = radius*np.cos(np.linspace(0, np.pi, n_training_samples_out))
    outer_circ_y = radius*np.sin(np.linspace(0, np.pi, n_training_samples_out))
    inner_circ_x = radius - radius*np.cos(np.linspace(0, np.pi, n_training_samples_in))
    inner_circ_y = -radius*np.sin(np.linspace(0, np.pi, n_training_samples_in))

    train_X = np.vstack(
        [np.append(outer_circ_x, inner_circ_x), np.append(outer_circ_y, inner_circ_y)]
    ).T
    y = np.hstack(
        [np.zeros(n_training_samples_out, dtype=np.intp), np.ones(n_training_samples_in, dtype=np.intp)]
    )

    if shuffle:
        train_X, y = util_shuffle(train_X, y, random_state=generator)

    if noise is not None:
        train_X += generator.normal(scale=noise, size=train_X.shape)

    return train_X, y

def multiview_moon(radius_list=[1.7,1.0,0.3], n_training_samples=1000, n_test_grid=100, n_ood_samples=200, random_state=42):
    """Make 2D multi-view dataset with Scikit-learn's moon dataset.
    The number of views is set to three.
    ----------
    radius_list: list or tuple, default=[1.7,1.0,0.3]
        Raiduses of circles for each view.
    n_training_samples: int or tuple of shape (2,), dtype=int, default=1000
        The number of points for each view.
    n_test_grid: int, default=100
        The number of test grids in each axis (train_X and train_Y). The total number of 
        test samples is n_test_grid^2.
    n_ood_samples: int, default=200
        The number of OOD samples.
    random_state: int, RandomState instance or None, default=42
        Determines random number generation for dataset shuffling and noise.

    Returns
    -------
    train_X : ndarray of shape (3*n_training_samples train_X, 2)
        The generated multi-view training samples.
    train_Y : ndarray of shape (3*n_training_samples train_X, 1)
        The generated multi-view training samples.
    test_X : ndarray of shape (n_test_grid^2, 2)
        The generated multi-view test samples.
    ood_X : ndarray of shape (n_ood_samples, 2)
        The generated multi-view OOD samples.
    """

    # Create 3 views separately with different radius_list of circles
    moon_noise = 0.13
    view1, y_1 = make_moons(n_training_samples, radius=radius_list[0], noise=moon_noise, random_state=random_state)
    view2, y_2 = make_moons(n_training_samples, radius=radius_list[1], noise=moon_noise, random_state=random_state)
    view3, y_3 = make_moons(n_training_samples, radius=radius_list[2], noise=moon_noise, random_state=random_state)

    # Make sure all the views share the same labels
    assert(np.sum(y_2!=y_3)==0)
    assert(np.sum(y_1!=y_3)==0)

    # Adjust the positions of samples a bit
    dist_apart = 0.08
    view2[y_2==0,1] -= dist_apart
    view2[y_2==1,1] += dist_apart
    view1[y_1==0,1] -= dist_apart
    view1[y_1==1,1] += dist_apart
    dist_apart = 0.14
    view3[y_3==0,1] -= dist_apart
    view3[y_3==1,1] += dist_apart
    view3[y_3==0,0] += dist_apart
    view3[y_3==1,0] -= dist_apart

    # Stack all the views into train_X and normalize it
    train_X = np.concatenate([view1,view2,view3], axis=0) 
    train_X = MinMaxScaler([0, 1]).fit_transform(train_X)
    train_Y = y_1

    # Test points
    margin = .4
    test_x = np.linspace(train_X[:,0].min()-margin, train_X[:,0].max()+margin, n_test_grid)
    test_y = np.linspace(train_X[:,1].min()-margin, train_X[:,1].max()+margin, n_test_grid)
    test_X = []
    for i in range(len(test_x)):
        for j in range(len(test_y)):
            test_X.append([test_x[i],test_y[j]])
    test_X = np.stack(test_X, axis=0)

    # OOD points
    noise_std = 0.04
    np.random.seed(random_state)
    ood_samples_x = np.clip(np.random.normal(loc=-.2, scale=noise_std, size=n_ood_samples), test_x.min(), test_x.max())
    ood_samples_y = np.clip(np.random.normal(loc=1.2, scale=noise_std, size=n_ood_samples), test_y.min(), test_y.max())
    ood_X = [ood_samples_x, ood_samples_y]

    return train_X, train_Y, test_X, ood_X