import logging
import os
import pickle
import sys
import numpy
from scipy.sparse import linalg

from numpy.ma import array
from image_handler import load_data, reconstruct_with_normalization
import scipy.linalg.blas


def eigen_face_construct(n_dim, scaling, scaling_factor):
    eigen_vectors = numpy.loadtxt("eigenvectors.csv", delimiter=',')
    for index in range(n_dim):
        try:
            name_ = str(index) + "th eigenface.bmp"
            reconstruct_with_normalization(eigen_vectors[index],
                                           scaling=scaling, scaling_factor=scaling_factor, name_of_file=name_)
        except IndexError:
            logging.warning("[" + str(index) + "] index of reconstruction is greater than dimension of eigenvectors")


def traverse_faces(rootdir_, scaling, scaling_factor):
    arrays = []
    means = []
    filenames = []
    for dirName, subdirList, fileList in os.walk(rootdir_, topdown=False):
        for fname in fileList:
            try:
                im_array = load_data(os.path.join(dirName, fname), scaling, scaling_factor)
                filenames.append(os.path.join(dirName, fname))
                means.append(im_array.mean())
                im_array = im_array - im_array.mean()  # center the data around 0
                if im_array.ndim < 2:
                    im_array = array(im_array, copy=False, subok=True, ndmin=2).T
                    arrays.append(im_array)
            except OSError:
                logging.warning("failed to load face:" + os.path.join(dirName, fname))
    return numpy.concatenate(arrays, 1), means, filenames


def create_covariance_matrix(rootdir_, scale_, r_):
    mat, means, files = traverse_faces(rootdir_, scaling=scale_, scaling_factor=r_)
    # cov = numpy.cov(mat)
    cov = scipy.linalg.blas.dgemm(alpha=1.0, a=mat, b=mat, trans_b=True)
    cov /= len(means)
    # THIS IS IMPORTANT!!!
    w, v = scipy.sparse.linalg.eigsh(cov, 100)  # we take first 100 biggest eigvalues with Lanczos algorithm
    # w, v = numpy.linalg.eigh(cov)

    # DO NOT DELETE THIS LINE!!!!
    idx = w.argsort()[::-1]
    w = w[idx]
    v = v[:, idx]
    # DO NOT DELETE THIS LINE!!!!

    # save artifacts
    numpy.savetxt("matrix.csv", mat, delimiter=",")
    numpy.savetxt("covariance.csv", cov, delimiter=",")
    numpy.savetxt("means.csv", means, delimiter=",")
    numpy.savetxt("eigenvalues.csv", w, delimiter=",")
    numpy.savetxt("eigenvectors.csv", v.T, delimiter=",")
    with open("files.txt", "wb") as fp:  # Pickling
        pickle.dump(files, fp)
        logging.debug(files)
    # log info for debug
    logging.debug("mat size is :")
    logging.debug(sys.getsizeof(mat))
    logging.debug("Covariance matrix created!")
    logging.debug(cov.shape)
    logging.debug("eigen values are :")
    logging.debug(w.shape)
    logging.debug("eigen vectors shape is:")
    logging.debug(v.shape)

    sum_ = 0
    for index, item in enumerate(w):
        sum_ = sum_ + item
        if sum_ * 1.0 / w.sum() > 0.95:
            logging.debug("effective size of human face PCA is :")
            logging.debug(index)
            return index

    raise Exception('Too few vectors taken add more degrees')


def reconstruct_to_n_degrees(degree_of_reconstruct, index_of_image_to_reconstruct,
                             scaling=False, scaling_factor=0.5):
    with open("files.txt", "rb") as fp:  # Unpickling
        eigen_files = pickle.load(fp)
    eigen_vectors = numpy.loadtxt("eigenvectors.csv", delimiter=',')
    means = numpy.loadtxt("means.csv", delimiter=',')
    original_im = load_data(eigen_files[index_of_image_to_reconstruct], scaling, scaling_factor)
    reconstruction = numpy.zeros(int(36000 * scaling_factor ** 2))
    for i in range(degree_of_reconstruct):
        reconstruction = reconstruction + numpy.dot(eigen_vectors[i], original_im.T) * eigen_vectors[i]
    return reconstruction + means[index_of_image_to_reconstruct]
