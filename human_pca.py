import logging

import numpy

from auxilary import eigen_face_construct, create_covariance_matrix

from image_handler import reconstruct_and_save


def reconstruct_to_n_degrees(degree_of_reconstruct, index_of_image_to_reconstruct):
    eigenfaces_array = numpy.loadtxt("eigen_faces.csv", delimiter=',')
    eigen_vectors = numpy.loadtxt("eigenvectors.csv", delimiter=',')
    reconstruction = numpy.zeros(36000)
    for i in range(degree_of_reconstruct):
        reconstruction = reconstruction + eigen_vectors[i][index_of_image_to_reconstruct] * eigenfaces_array[i]
    print(reconstruction)
    return reconstruction


if __name__ == '__main__':
    # Setup
    logging.basicConfig(filename='human_pca.log', level=logging.DEBUG)
    rootdir = './faces94/female/slbirc'
    r = 1. / 10  # scaling factor that doesn't make my computer explode
    scale = True

    # test
    create_covariance_matrix(rootdir, scale, r)
    n = 10
    eigen_face_construct(n)
    # save_eigenface_images_to_disk()
    m = 0
    im = reconstruct_to_n_degrees(n, m)
    reconstruct_and_save(im, scaling=False, name_of_file=str(m) + "th_photo_from_" + str(n) + "_eigenfaces.jpeg")

    # TODO : normalize
    # TODO : dot product m vectors to xi;
