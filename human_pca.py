import logging

from auxilary import reconstruct_to_n_degrees, create_covariance_matrix, eigen_face_construct
from image_handler import reconstruct_with_normalization, load_data

if __name__ == '__main__':
    # Setup
    logging.basicConfig(filename='human_pca.log', level=logging.DEBUG)
    rootdir = './faces94'
    logging.debug("working on dir: " + rootdir)
    scale = True
    r = 5. / 10  # scaling factor that doesn't make my computer explode
    logging.debug("scaling is: " + str(scale))
    if scale:
        logging.debug("and scale is: " + str(r))

    # test
    effective_dim = 10
    effective_dim = create_covariance_matrix(rootdir, scale, r)
    m = 0
    m_ = load_data('lfso.1.jpg', scaling=True, scaling_factor=r)
    reconstruct_with_normalization(m_, scaling=True, scaling_factor=r, name_of_file="temp.jpg")
    eigen_face_construct(effective_dim, scaling=scale, scaling_factor=r)
    im = reconstruct_to_n_degrees(effective_dim, m, scaling=scale, scaling_factor=r)
    reconstruct_with_normalization(im, scaling=scale, scaling_factor=r,
                                   name_of_file=str(m) + "th_photo_from_" + str(effective_dim) + "_eigenfaces.jpeg")

