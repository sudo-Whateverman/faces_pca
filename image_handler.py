from PIL import Image, ImageOps, ImageEnhance
import numpy
import scipy.misc


def load_data(filename_, scaling, scaling_factor):
    image_ = Image.open(filename_).convert('F')  # Load image as grayscale.
    data_arr_ = numpy.asarray(image_)  # original non-flat array is (200, 180) shaped, flat is 36000
    if scaling:
        data_arr_ = scipy.misc.imresize(data_arr_, scaling_factor, interp='bilinear', mode=None)
    return data_arr_.flatten()


def reconstruct_with_normalization(image, scaling, scaling_factor, name_of_file="temp"):
    # reconstruct from array
    if scaling:
        im_ = Image.fromarray(image.reshape((int(200 * scaling_factor), int(180 * scaling_factor))))
    else:
        im_ = Image.fromarray(numpy.uint8(image.reshape(200, 180)))
    im_ = ImageOps.autocontrast(im_, cutoff=0, ignore=None)
    if im_.mode != 'RGB':
        im_ = im_.convert('RGB')
    im_.save(name_of_file)
