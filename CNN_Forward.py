import numpy as np


def Convolution_Forward(image, _filter, bias, stride):  # Convolve 'filter' over 'image' using 'stride'

    img_depth, img_dim, _ = image.shape  # image dimensions
    (n_f, f_depth, f_dim, _) = _filter.shape  # filter dimensions
    out_dim = int((img_dim - f_dim) / stride) + 1  # output dimensions
    assert img_depth == f_depth, "Input image's depth must equal with filter's depth"
    out = np.zeros((n_f, out_dim, out_dim))  # Initialise output that holds the values of the convolution process

    for current_f in range(n_f):  # Convolve 'filter' over every part of 'image', adding the bias at each step
        current_y = out_y = 0
        while current_y + f_dim <= img_dim:  # Slide filter vertically across the image
            current_x = out_x = 0
            while current_x + f_dim <= img_dim:  # Slide filter horizontally across the image
                # Perform convolution and add the bias
                out[current_f, out_y, out_x] = np.sum(_filter[current_f] * image[:, current_y: current_y + f_dim,
                                                                    current_x: current_x + f_dim]) + bias[current_f]
                current_x += stride;    out_x += 1
            current_y += stride;    out_y += 1

    return out


def Maxpooling_Forward(image, kernel_size, stride):  # Downsampling 'image' using 'kernel_size' and 'stride'

    i_depth, i_height, i_width = image.shape  # image dimensions

    # Output dimensions after maxpooling
    o_height = int((i_height - kernel_size) / stride) + 1
    o_width = int((i_width - kernel_size) / stride) + 1

    out = np.zeros((i_depth, o_height, o_width))  # Initialise output that holds the values of the maxpooling process

    # Slide maxpooling window over every part of 'image' using 'stride' and get max value at each step
    for i in range(i_depth):
        current_y = out_y = 0
        while current_y + kernel_size <= i_height:  # Slide window vertically across the image
            current_x = out_x = 0
            while current_x + kernel_size <= i_width:  # Slide window horizontally across the image
                # Obtain max value within the window at each step
                out[i, out_y, out_x] = np.max(image[i, current_y: current_y + kernel_size,
                                                      current_x: current_x + kernel_size])
                current_x += stride;    out_x += 1
            current_y += stride;    out_y += 1

    return out


def Softmax_Activation(X):  # Maps all final dense layer outputs to a vector whose elements sum up to 1
    out = np.exp(X)
    return out / np.sum(out)  # Divide the exponentiated vector by its sum


# Loss function assigns a real-valued number to define the model’s accuracy when predicting the output
def Categorical_Cross_Entropy(probability, label):
    # Multiply the desired output label by the log of the prediction, then sum all values in the vector
    return -np.sum(label * np.log(probability))


def Image_Label(image_name):  # Create a label for each image
    i_label = image_name.split("_", 1)[0]
    if 'Aeroplane' in i_label:  # Assign '0' for 'Aeroplane' images
        return 0
    elif 'Helicopter' in i_label:  # Assign '1' for 'Helicopter' images
        return 1
