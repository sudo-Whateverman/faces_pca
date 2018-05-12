import logging
import os
import pickle
import sys
import numpy
from numpy import linalg
from numpy.ma import array
from image_handler import load_data


def eigen_face_construct(n_dim):
    eigen_vectors = numpy.loadtxt("eigenvectors.csv", delimiter=',')
    # means = numpy.loadtxt("means.csv", delimiter=",")
    eigen_faces = []
    with open("files.txt", "rb") as fp:  # Unpickling
        eigen_files = pickle.load(fp)
    for i in range(n_dim):
        try:
            reconstruction = numpy.zeros(36000)
            for index, item in enumerate(eigen_vectors[i]):
                reconstruction = reconstruction + item * load_data(eigen_files[index], scaling=False)
            reconstruction = array(reconstruction, copy=False, subok=True, ndmin=2).T
            eigen_faces.append(reconstruction)
        except IndexError:
            logging.warning("[" + str(i) + "] index of reconstruction is greater than dimension of eigenvectors")
    numpy.savetxt("eigen_faces.csv", eigen_faces, delimiter=",")


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
                im_array = im_array - im_array.mean()  # center the data around 0 TODO: normalize here
                if im_array.ndim < 2:
                    im_array = array(im_array, copy=False, subok=True, ndmin=2).T
                    arrays.append(im_array)
            except OSError:
                logging.warning("failed to load face:" + os.path.join(dirName, fname))
    return numpy.concatenate(arrays, 1), means, filenames


def create_covariance_matrix(rootdir_, scale_, r_):
    mat, means, files = traverse_faces(rootdir_, scaling=scale_, scaling_factor=r_)
    cov = numpy.cov(mat)
    w, v = linalg.eigh(cov)  # since the covariance matrix is symmetrical and hermitian, only real values will be given
    # sort eigenvalues and vectors
    idx = w.argsort()[::-1]
    w = w[idx]
    v = v[:, idx]
    # save artifacts
    numpy.savetxt("matrix.csv", mat, delimiter=",")
    numpy.savetxt("covariance.csv", cov, delimiter=",")
    numpy.savetxt("means.csv", means, delimiter=",")
    numpy.savetxt("eigenvalues.csv", w, delimiter=",")
    numpy.savetxt("eigenvectors.csv", v, delimiter=",")
    with open("files.txt", "wb") as fp:  # Pickling
        pickle.dump(files, fp)
        logging.debug(files)
    # log info for debug
    logging.debug("mat size is :")
    logging.debug(sys.getsizeof(mat))
    logging.debug("Covariance matrix created!")