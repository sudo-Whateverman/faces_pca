import os
from PIL import Image
import numpy
import logging

def load_data(filename_):
    im = Image.open(filename_).convert('L')  # Load image as grayscale.
    data_arr_ = numpy.asarray(im.getdata())  # non-flat array is (200, 180) shaped, flat is 36000
    return data_arr_

def traverse_faces():
    rootdir = './faces94'
    for dirName, subdirList, fileList in os.walk(rootdir, topdown=False):
        for fname in fileList:
            try:
                load_data(os.path.join(dirName, fname))
            except OSError:
                logging.warning("failed to load face:" + os.path.join(dirName, fname))

def create_covariance_matrix():
    pass

def sort_eigenvectors():
    pass

def approxiface(n_dim):
    pass

def eigenface(n_dim):
    pass


if __name__ == '__main__':
    # test 1
    # filename = "./faces94/female/9336923/9336923.1.jpg"
    # data_arr = load_data(filename)
    # print(data_arr.shape)

    # test 2
    logging.basicConfig(filename='human_pca.log', level=logging.DEBUG)
    traverse_faces()