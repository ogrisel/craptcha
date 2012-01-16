"""Interactive script to evaluate models suited for the task

This is mean to be run in an ``ipython --pylab`` session
using the ``%run model_selection`` magic.
"""

import pylab as pl
import numpy as np
from sklearn.utils import shuffle
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from craptcha import samples

ts = samples.TrainingSet()
ts.add(samples.samples["easy"])

pil_images, labels = zip(*ts.getPairs())  # argl! not PEP8

# convert PIL images to numpy array
images = []
for image in pil_images:
    images.append(np.array(image.getdata(), dtype=np.float))
images = np.array(images) / 255.  # original pixel data are in range 0-255

# convert symbol names to integer target
target_names = np.unique(labels)
target_names.sort()
name_to_idx = dict((n, i) for i, n in enumerate(target_names))
target = np.array([name_to_idx[label] for label in labels], dtype=np.int)


# shuffle the data to avoid structured splits
images, target = shuffle(images, target, random_state=42)

# build a SVM classifier and look for the best parameters using grid search
# and cross validation accross the dataset (we should use another
# independently generated evaluation set for the final model evaluation by the
# way)

clf = SVC()
grid = [
    {
        'kernel': ['rbf'],
        'gamma': [1e-1, 1e-2, 1e-3, 1e-4],
        'C': [1, 10, 100, 1000]
    },
    {
        'kernel':['linear'],
        'C': [1, 10, 100, 1000],
    }
]

# TODO: Uncomment this onces the dataset is big enough
#gs = GridSearchCV(clf, grid, verbose=1, n_jobs=-1, cv=3).fit(images, target)
#print gs

# visual inspection of some of the input images
n_rows, n_cols = 5, 10
pl.figure(figsize=(1.8 * n_cols, 2.4 * n_rows))
for i, image, target in zip(range(n_rows * n_cols), images, target):
    pl.subplot(n_rows, n_cols, i + 1)
    pl.imshow(image.reshape((25, 25)))
    pl.imshow(image.reshape((25, 25)), cmap=pl.cm.gray_r,
              interpolation='nearest')
    pl.title(target_names[target])
    pl.xticks(())
    pl.yticks(())

pl.show()
