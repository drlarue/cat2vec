"""Data and model setup for cat2vec

1. Generate the feature vectors of all the available cats (stacked into a matrix)
2. Get the pre-trained cat weights for the final predictions layer
"""
import numpy as np
import scipy.sparse as sp
import h5py

import os
import glob

from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.models import Model

m_vgg16 = VGG16(weights='imagenet', include_top=True)
m_vgg16_fc2 = Model(inputs=m_vgg16.input, outputs=m_vgg16.get_layer('fc2').output)

# MAKE SURE THESE ARE IDENTICAL TO `cat2vec.py`
pic_dir = 'petfinder/' # directory of petfinder cats
pic_names = np.sort(os.listdir(pic_dir)) # list of sorted filenames
cat_pics = [pic_dir + f for f in pic_names] # list of filepaths


def generate_features_vgg16(save_as, pic_paths=cat_pics, model=m_vgg16_fc2, n=4096):
    """Construct a feature matrix, where each image's feature vector is inserted one row
    at a time.

    Arguments:
        * save_as: name by which the feature matrix is to be saved (e.g., 'f_matrix')
        * pic_paths: list of image filepaths
        * model: keras model used for feature extraction
        * n: number of features (columns)
    """
    preds = sp.lil_matrix((len(pic_paths) + 1, n))
    # vgg16: 4096 for the fc2 (penultimate) layer, 25088 if include_top=False

    for i, f in enumerate(pic_paths):
        print(i+1)
        img = image.load_img(f, target_size=(224, 224))
        x_raw = image.img_to_array(img)
        x_expand = np.expand_dims(x_raw, axis=0)
        x = preprocess_input(x_expand)
        pred = model.predict(x)
        preds[i+1, :] = pred.ravel() # leave i=0 for the input image

    # can't save in LIL format; convert to CSR format to save
    sp.save_npz('{}.npz'.format(save_as), preds.tocsr())


def save_cat_wts(arch):
    """Save the cat weights for the final predictions layer as a dictionary.

    Argument:
        * arch: the name of the architecture; either 'inception' or 'vgg16'
    """
    # weights to specific cat neurons:
        # Imagenet index
            # 281: tabby
            # 282: tiger_cat
            # 283: Persian_cat
            # 284: Siamese_cat
            # 285: Egyptian_cat

    if arch == 'inception':
        weights_path = os.path.join(os.environ.get('HOME'),
                                    '.keras/models/inception_v3_weights_tf_dim_ordering_tf_kernels.h5')
        wts_file = h5py.File(weights_path, 'r')
        final_wts = wts_file.get('predictions').get('predictions').get('kernel:0')

    elif arch == 'vgg16':
        weights_path = os.path.join(os.environ.get('HOME'),
                                    '.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels.h5')
        wts_file = h5py.File(weights_path, 'r')
        wts_file.get('predictions').get('predictions_W_1:0')
        final_wts = wts_file.get('predictions').get('predictions_W_1:0')

    cat_wts = np.array(final_wts)[:, 281:286]
    wts_file.close()
    np.save('cat_wts.npy', cat_wts)
