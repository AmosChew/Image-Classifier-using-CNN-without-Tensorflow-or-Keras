import numpy as np


def Convolution_Backward(dconv_prev, conv_in, _filter, stride):  # Backpropagation through a convolutional layer

    (n_f, f_depth, f_dim, _) = _filter.shape  # Filter dimensions
    (_, conv_dim, _) = conv_in.shape  # Convoluted input dimensions

    # Initialize Derivatives
    dout = np.zeros(conv_in.shape)
    dfilter = np.zeros(_filter.shape)
    dbias = np.zeros((n_f, 1))

    for current_f in range(n_f):  # Loop through all filters
        current_y = out_y = 0
        while current_y + f_dim <= conv_dim:  # Slide window vertically across the convoluted input
            current_x = out_x = 0
            while current_x + f_dim <= conv_dim:  # Slide window horizontally across the convoluted input
                # Loss gradient of the filter
                dfilter[current_f] += dconv_prev[current_f, out_y, out_x] * conv_in[:, current_y: current_y + f_dim,
                                                                            current_x: current_x + f_dim]
                # Loss gradient of the input to the convolution process
                dout[:, current_y: current_y + f_dim, current_x: current_x + f_dim] += dconv_prev[current_f, out_y,
                                                                                            out_x] * _filter[current_f]
                current_x += stride;    out_x += 1
            current_y += stride;    out_y += 1

        # Loss gradient of the bias
        dbias[current_f] = np.sum(dconv_prev[current_f])

    return dout, dfilter, dbias


def nanargmax(array):  # Return index of the largest non-nan value in the array
    index = np.nanargmax(array)
    _index = np.unravel_index(index, array.shape)  # Converts array of flat indices into a tuple of coordinate arrays
    return _index


def Maxpooling_Backward(dpool, orig, kernel_size, stride):  # Backpropagation through a maxpooling layer

    (orig_depth, orig_dim, _) = orig.shape  # Maxpooled input dimensions
    dout = np.zeros(orig.shape)  # Initialize output derivatives

    for current_c in range(orig_depth):
        current_y = out_y = 0
        while current_y + kernel_size <= orig_dim:  # Slide window vertically across the maxpooled input
            current_x = out_x = 0
            while current_x + kernel_size <= orig_dim:  # Slide window horizontally across the maxpooled input
                # Obtain index of largest value in input for current window
                (x, y) = nanargmax(orig[current_c, current_y: current_y + kernel_size,
                                   current_x: current_x + kernel_size])
                # Gradients are passed through the indices of largest value in original maxpooling during forward step
                dout[current_c, current_y + x, current_x + y] = dpool[current_c, out_y, out_x]

                current_x += stride;    out_x += 1
            current_y += stride;    out_y += 1

    return dout
