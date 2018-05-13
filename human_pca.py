import logging

from auxilary import reconstruct_to_n_degrees, create_covariance_matrix, eigen_face_construct, average_face_reconstruct, \
    traverse_faces
from image_handler import reconstruct_with_normalization, load_data

if __name__ == '__main__':
    # Setup
    logging.basicConfig(filename='human_pca.log', level=logging.DEBUG)
    rootdir = './faces94/female'
    logging.debug("working on dir: " + rootdir)
    scale = False
    r = 5. / 10  # scaling factor that doesn't make my computer explode
    logging.debug("scaling is: " + str(scale))
    if scale:
        logging.debug("and scale is: " + str(r))

    # sanity check!
    m_ = load_data('./faces94/female/lfso/lfso.1.jpg', scaling=scale, scaling_factor=r)
    reconstruct_with_normalization(m_, scaling=scale, scaling_factor=r, name_of_file="temp.jpg", normalize=False)

    # test
    effective_dim = 300
    effective_dim = create_covariance_matrix(rootdir, scale, r)
    mat, means, files = traverse_faces(rootdir, scaling=scale, scaling_factor=r)

    average_face_reconstruct(scale, r)
    m = 4
    print_eigenvectors = True
    if print_eigenvectors:
        eigen_face_construct(10, scaling=scale, scaling_factor=r)
    degrees = [1, 10, 30, effective_dim]
    for degree in degrees:
        im = reconstruct_to_n_degrees(degree, m, scaling=scale, scaling_factor=r)
        image = reconstruct_with_normalization(im, scaling=scale, scaling_factor=r,
                                   name_of_file=str(m) + "th_photo_from_" + str(degree) + "_eigenfaces.jpeg")
