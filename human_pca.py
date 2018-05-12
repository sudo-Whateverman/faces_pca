import logging

from auxilary import reconstruct_to_n_degrees, create_covariance_matrix
from image_handler import reconstruct_and_save, reconstruct_with_normalization

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
    # effective_dim = create_covariance_matrix(rootdir, scale, r)
    n = 50
    m = 0
    im = reconstruct_to_n_degrees(n, m, scaling=scale, scaling_factor=r)
    reconstruct_with_normalization(im, scaling=scale, scaling_factor=r,
                         name_of_file=str(m) + "th_photo_from_" + str(n) + "_eigenfaces.jpeg")

