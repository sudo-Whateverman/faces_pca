from PIL import Image
import PIL
import numpy


def load_data(filename_, scaling=False, scaling_factor=.1):
    image_ = Image.open(filename_).convert('L')  # Load image as grayscale.
    if scaling:
        image_ = image_.resize((int(200 * scaling_factor), int(180 * scaling_factor)))  # rescale to make conv possible
    data_arr_ = numpy.asarray(image_.getdata())  # original non-flat array is (200, 180) shaped, flat is 36000
    return data_arr_


def reconstruct_and_save(data_arr_, scaling=True, scaling_factor=.1, name_of_file="temp"):
    # reconstruct from array
    if scaling:
        im_ = Image.fromarray(numpy.uint8(data_arr_.reshape((int(200 * scaling_factor), int(180 * scaling_factor)))))
    else:
        im_ = Image.fromarray(numpy.uint8(data_arr_.reshape(200, 180)))
    im_.save(name_of_file)


def save_eigenface_images_to_disk():
    eigenfaces_array = numpy.loadtxt("eigen_faces.csv", delimiter=',')

    for index, face in enumerate(eigenfaces_array):
        reconstruct_and_save(eigenfaces_array[index], scaling=False, name_of_file="eigenface" + str(index) + ".jpeg")


def eigen_face_construct_from_subsamples(image, namefile):
    PIL.ImageOps.autocontrast(image, cutoff=0, ignore=None)
    Image.save(namefile)