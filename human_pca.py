import os
from PIL import Image
import numpy
import logging
from numpy.ma import array
import sys
from numpy import linalg
import pickle


def load_data(filename_, scaling=False, scaling_factor=.1):
    im = Image.open(filename_).convert('L')  # Load image as grayscale.
    if scaling:
        im = im.resize((int(200 * scaling_factor), int(180 * scaling_factor)))  # rescale to make conv possible
    data_arr_ = numpy.asarray(im.getdata())  # original non-flat array is (200, 180) shaped, flat is 36000
    return data_arr_


def traverse_faces(rootdir_, scaling, scaling_factor):
    arrays = []
    means = []
    filenames = []
    for dirName, subdirList, fileList in os.walk(rootdir_, topdown=False):
        for fname in fileList:
            try:
                # update the mat with column stack
                # add mean = 0 values only
                im_array = load_data(os.path.join(dirName, fname), scaling, scaling_factor)
                filenames.append(os.path.join(dirName, fname))
                means.append(im_array.mean())
                im_array = im_array - im_array.mean()
                if im_array.ndim < 2:
                    im_array = array(im_array, copy=False, subok=True, ndmin=2).T
                    arrays.append(im_array)
            except OSError:
                logging.warning("failed to load face:" + os.path.join(dirName, fname))
    return numpy.concatenate(arrays, 1), means, filenames


def reconstruct_and_show(data_arr_, scaling=True, scaling_factor=.1, name_of_file="temp"):
    # reconstruct from array
    if scaling:
        im_ = Image.fromarray(numpy.uint8(data_arr_.reshape((int(200 * scaling_factor), int(180 * scaling_factor)))))
    else:
        im_ = Image.fromarray(numpy.uint8(data_arr_.reshape(200, 180)))
    im_.save(name_of_file)


def create_covariance_matrix(rootdir_, scale_, r_):
    mat, means, files = traverse_faces(rootdir_, scaling=scale_, scaling_factor=r_)
    cov = numpy.cov(mat.T)

    numpy.savetxt("matrix.csv", mat, delimiter=",")
    numpy.savetxt("covariance.csv", cov, delimiter=",")
    numpy.savetxt("means.csv", means, delimiter=",")

    w, v = linalg.eigh(cov)
    # sort eigenvalues and vectors
    idx = w.argsort()[::-1]
    w = w[idx]
    v = v[:, idx]
    numpy.savetxt("eigenvalues.csv", w, delimiter=",")
    numpy.savetxt("eigenvectors.csv", v, delimiter=",")
    with open("files.txt", "wb") as fp:  # Pickling
        pickle.dump(files, fp)
        logging.debug(files)
    log_txt = "mat size is :"
    logging.debug(log_txt)
    logging.debug(sys.getsizeof(mat))


def eigen_face_construct(n_dim):
    eigen_vectors = numpy.loadtxt("eigenvectors.csv", delimiter=',')
    means = numpy.loadtxt("means.csv", delimiter=",")
    eigen_faces = []
    with open("files.txt", "rb") as fp:  # Unpickling
        eigen_files = pickle.load(fp)
    for i in range(n_dim):
        try:
            reconstruction = numpy.zeros(36000)
            for index, item in enumerate(eigen_vectors[i]):
                reconstruction = reconstruction + item * load_data(eigen_files[index], scaling=False) + means[i]
            reconstruction = array(reconstruction, copy=False, subok=True, ndmin=2).T
            eigen_faces.append(reconstruction)
        except IndexError:
            logging.warning("[" + str(i) + "] index of reconstruction is greater than dimension of eigenvectors")
    return eigen_faces


def reconstruct_to_n_degrees(n, m, eigenfaces):
    eigen_vectors = numpy.loadtxt("eigenvectors.csv", delimiter=',')
    for i in range(n):

    pass


if __name__ == '__main__':
    # Setup
    logging.basicConfig(filename='human_pca.log', level=logging.DEBUG)
    rootdir = './faces94'
    r = 4. / 10  # scaling factor that doesn't make my computer explode
    scale = False

    # test
    create_covariance_matrix(rootdir, scale, r)
    n = 10
    eigenfaces = eigen_face_construct(n)
    for index, face in enumerate(eigenfaces):
        reconstruct_and_show(eigenfaces, scaling=False, name_of_file="eigenface" + str(index) + ".jpeg")

    reconstruct_to_n_degrees(n, m=0, eigenfaces)
