from CNN_Forward import *
import numpy as np
import pickle
import imageio
import glob
from tqdm import tqdm


# Make predictions with trained filters/weights
def Predict(image, _f1, _f2, _w3, _w4, _b1, _b2, _b3, _b4, c_stride=1, kernel_size=2, mp_stride=2):

    convolve_1 = Convolution_Forward(image, _f1, _b1, c_stride)  # 1st Convolution
    convolve_1[convolve_1 <= 0] = 0  # pass through ReLU Layer
    pooled_1 = Maxpooling_Forward(convolve_1, kernel_size, mp_stride)  # 1st Maxpooling

    convolve_2 = Convolution_Forward(pooled_1, _f2, _b2, c_stride)  # 2nd Convolution
    convolve_2[convolve_2 <= 0] = 0  # pass through ReLU Layer
    pooled_2 = Maxpooling_Forward(convolve_2, kernel_size, mp_stride)  # 2nd Maxpooling

    (p2_f, p2_dim, _) = pooled_2.shape
    Z1 = pooled_2.reshape((p2_f * p2_dim * p2_dim, 1))  # Flatten Maxpooled Layer
    Z = _w3.dot(Z1) + _b3  # 1st Dense Layer
    Z[Z <= 0] = 0  # pass through ReLU Layer
    out = _w4.dot(Z) + _b4  # 2nd Dense Layer

    _probability = Softmax_Activation(out)  # Predict class probability

    return np.argmax(_probability), np.max(_probability)


# Extract parameters from trained Model
parameters, cost = pickle.load(open('Aeroplane_Helicopter_v6.pkl', 'rb'))
[f1, f2, w3, w4, b1, b2, b3, b4] = parameters

# Obtain test images
img_input = []
file_aero = glob.glob("Aeroplane/Test_1/*.jpg")
file_heli = glob.glob("Helicopter/Test_1/*.jpg")
for fileA in file_aero:
    input_imageA = imageio.imread(fileA, as_gray=True)
    img_input.append([input_imageA, Image_Label(fileA)])
for fileH in file_heli:
    input_imageH = imageio.imread(fileH, as_gray=True)
    img_input.append([input_imageH, Image_Label(fileH)])

X = np.array([i[0] for i in img_input]).reshape(80, 1, 120, 120)  # Stores image data
X -= int(np.mean(X));   X /= int(np.std(X))  # Normalize the data
_labels = [i[1] for i in img_input]  # Stores image labels

correct = 0
print()
print("\nComputing accuracy over test set:")
t = tqdm(range(len(X)), leave=True)

for i in t:
    _X = X[i]
    prediction, probability = Predict(_X, f1, f2, w3, w4, b1, b2, b3, b4)
    if prediction == _labels[i]:
        correct += 1  # Count the number of correct predictions

    t.set_description("Accuracy:%0.2f%%" % (float(correct / (i + 1)) * 100))

print("Overall Accuracy: %.2f" % (float(correct / len(X) * 100)))
