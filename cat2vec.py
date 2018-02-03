import numpy as np
import pandas as pd
import scipy.sparse as sp

import os
import glob

from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model


class Cat2Vec:

    def __init__(self):
        # MAKE SURE THESE ARE IDENTICAL TO `cat2vec_setup.py`
        self.pic_dir = 'petfinder/' # directory of petfinder cats
        self.pic_names = np.sort(os.listdir(self.pic_dir)) # list of sorted filenames
        self.cat_pics = [self.pic_dir + f for f in self.pic_names] # list of filepaths

        # full model
        self.m_vgg16 = VGG16(weights='imagenet', include_top=True)

        # model with the predictions (final) layer removed
        self.m_vgg16_fc2 = Model(inputs=self.m_vgg16.input,
                                 outputs=self.m_vgg16.get_layer('fc2').output)

        # load saved feature matrix of petfinder cats (input image not added)
        # matrix saved as CSR; load it as array
        # see `cat2vec_setup.py` for how I got this
        self.headless_fm = sp.load_npz('f_vgg16_fc2.npz').toarray()

        # calculate the norms of the headless (input image missing) feature matrix
        self.headless_norms = np.linalg.norm(self.headless_fm, axis=1)

        # load pre-trained weights specific to the cat outputs
        # 5 cat categories in Imagenet, so shape is (4096, 5)
        # see `cat2vec_setup.py` for how I got this
        self.cat_wts = np.load('cat_wts_VGG16.npy')


    def predict(self, model, input_img, top_n=10):
        """Provide the top n predictions (from the full model) for the input image.

        Arguments:
            * model: full Imagenet model
            * input_img: input image filepath
        """
        img = image.load_img(input_img, target_size=(224, 224))
        x_raw = image.img_to_array(img)

        if x_raw.shape[-1] == 4: # png transparency channel
            x_raw = x_raw[:, :, :-1]

        x_expand = np.expand_dims(x_raw, axis=0)
        x = preprocess_input(x_expand)
        preds = model.predict(x)

        return decode_predictions(preds, top=top_n)[0]


    def imagenet_cat(self, decoded_probs):
        """Normalize the input image's predicted probabilities of the cat categories.

        I.e., predicted probabilities for the 5 cat categories sum up to 1 (0 if no cat predicted).

        Argument:
            * decoded_probs: the top n predicted probabilities (output from the `predict` function)
        """
        decoded_dict = {pr[1]:pr[2] for pr in decoded_probs}

        # 5 Imagenet cat outputs & their corresponding indexes
        cat_key = ['tabby', 'tiger_cat', 'Persian_cat', 'Siamese_cat', 'Egyptian_cat']
        cat_idx = [281, 282, 283, 284, 285]

        cat_dict = {}
        for key, idx in zip(cat_key, cat_idx):
            if key in decoded_dict:
                cat_dict[idx] = decoded_dict[key]
            else:
                cat_dict[idx] = 0

        breed_importances = np.empty(5)
        for key in cat_dict: # loop through keys -- no need to specify .keys()
            breed_importances[key - cat_idx[0]] = cat_dict[key]

        cat_sum = np.sum(breed_importances)

        if cat_sum == 0:
            return "Hmm, that doesn't look like a cat..."

        else:
            return breed_importances / cat_sum


    def add_input(self, model, input_img):
        """Vectorize the input image.

        Arguments:
            * model: model used for feature extraction
            * input_img: input image filepath
        """
        img = image.load_img(input_img, target_size=(224, 224))
        x_raw = image.img_to_array(img)
        x_expand = np.expand_dims(x_raw, axis=0)
        x = preprocess_input(x_expand)
        pred = model.predict(x)

        return pred


    def generate_weighted_features_v2(self, full_feature_matrix, rel_breed_importances):
        """Take the max relative breed importance and apply only those cat category weights.

        Arguments:
            * full_feature_matrix: feature matrix of ALL cat vectors (input image included)
            * rel_breed_importances: normalized predicted probs of cat categories
                                     (output from the `imagenet_cat` function)
        """
        max_breed_idx = np.argmax(rel_breed_importances)
        wcat = self.cat_wts[:, max_breed_idx]

        # stack vectors (need same dimensions as feature matrix)
        Wcat = np.tile(wcat, (full_feature_matrix.shape[0], 1))

        return np.multiply(full_feature_matrix, Wcat)


    def cosine_similarity(self, M, N):
        """Calculate cosine similarity between input and rest of cat vectors.

        Arguments:
            * M: full feature matrix
            * N: row vector norms of the full feature matrix
        """
        input_vector = M[0, :]
        input_vector_ = np.tile(input_vector, (M.shape[0], 1))

        return (np.sum(M * input_vector_, axis=1) /
                (N * np.linalg.norm(input_vector)))


    def unique_matching_cats(self, sim, pic_fpath_list):
        """
        Arguments:
            * sim: vector of cosine similarities (output of `cosine_similarity` function)
            * pic_fpath_list: list of image filepaths
        """
        # similarities with input image; exclude similarity with itself
        sim_input = sim[1:]

        # top 12 sim values' corresponding indexes
        top_index = np.argpartition(sim_input, -12)[-12:]
        # top sim values
        top_sims = sim_input[top_index]

        df = pd.DataFrame(data=top_sims.T, columns=['sim'], index=top_index.T)
        df.sort_values('sim', ascending=False, inplace=True)

        for im_idx in df.index:
            pic_id_sub = pic_fpath_list[im_idx].split('/')[-1].split('.')[-2]
            df.loc[im_idx, 'pic_id_sub'] = pic_id_sub
            df.loc[im_idx, 'pic_id'] = pic_id_sub.split('_')[0]

        return df.drop_duplicates(subset=['pic_id'], keep='first')


    def get_cats(self, input_img, use_weighted=False):
        """Get top cat matches

        Arguments:
            * input_img: input image filepath
            * use_weighted: whether to use weighted vectors or not

        Return:
            * DataFrame of the top matching cats with the following columns:
                index: index of the picture in the filepath list
                sim: similarity score
                pic_id_sub: petfinder cat id + picture id
                pic_id: petfinder cat id
        """
        decoded_probs = self.predict(self.m_vgg16, input_img)
        rel_breed_imp = self.imagenet_cat(decoded_probs)

        if isinstance(rel_breed_imp, str):
            return rel_breed_imp # this returns string, "Hmm, that doesn't look like a cat..."

        else:
            input_vector = self.add_input(self.m_vgg16_fc2, input_img)
            self.headless_fm[0, :] = input_vector # add input vector in 1st row
            self.headless_norms[0] = np.linalg.norm(input_vector) # add input vector norm in 1st row

            if use_weighted is False:
                matrix = self.headless_fm
            elif use_weighted is True:
                matrix = self.generate_weighted_features_v2(self.headless_fm, rel_breed_imp)

            sim = self.cosine_similarity(matrix, self.headless_norms)

            df_top_matches = self.unique_matching_cats(sim, self.cat_pics)

            return df_top_matches
