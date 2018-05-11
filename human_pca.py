import os
from PIL import Image
import numpy
import logging
from numpy.ma import array


def load_data(filename_):
    im = Image.open(filename_).convert('L')  # Load image as grayscale.
    data_arr_ = numpy.asarray(im.getdata())  # non-flat array is (200, 180) shaped, flat is 36000
    return data_arr_

def traverse_faces(rootdir_):
    arrays = []
    for dirName, subdirList, fileList in os.walk(rootdir_, topdown=False):
        for fname in fileList:
            try:
                # update the mat with column stack
                # add mean = 0 values only
                im_array = load_data(os.path.join(dirName, fname))
                im_array = im_array - im_array.mean()
                if im_array.ndim < 2:
                    im_array = array(im_array, copy=False, subok=True, ndmin=2).T
                    arrays.append(im_array)
            except OSError:
                logging.warning("failed to load face:" + os.path.join(dirName, fname))
    return numpy.concatenate(arrays, 1)


def reconstruct_and_show(data_arr_):
    # reconstruct from array
    im_ = Image.fromarray(numpy.uint8(data_arr_.reshape((200, 180))))
    im_.show()

def create_covariance_matrix():
    pass

def sort_eigenvectors():
    pass

def approxiface(n_dim):
    pass

def eigenface(n_dim):
    pass


if __name__ == '__main__':
    # Setup
    logging.basicConfig(filename='human_pca.log', level=logging.DEBUG)
    rootdir_ = './faces94'

    # test 1
    # filename = "./faces94/female/9336923/9336923.1.jpg"
    # data_arr = load_data(filename)
    # print(data_arr - data_arr.mean())
    #reconstruct_and_show(data_arr - data_arr.mean())




    # test 2
    # traverse_faces()

    # # test 3
    # filename = "./faces94/female/9336923/9336923.1.jpg"
    # filename2 = "./faces94/female/9336923/9336923.1.jpg"
    # filename3 = "./faces94/female/9336923/9336923.1.jpg"
    #
    # data_arr = load_data(filename)
    # data_arr2 = load_data(filename2)
    # data_arr3 = load_data(filename3)
    #
    # mat = numpy.column_stack((data_arr, data_arr2))
    # mat = numpy.column_stack((mat, data_arr3))
    # print(mat)

    # test 3
    mat = traverse_faces('./faces94/female/anpage')
    # that gives a Memmory Error... oops?!
    var_matrix = numpy.cov(mat)
    print(var_matrix.shape)
