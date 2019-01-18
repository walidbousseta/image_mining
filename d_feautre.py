import mahotas
import cv2
import numpy as np
import matplotlib.pyplot as plt

def d_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature


def d_haralick(image):    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick
 

def d_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    
    return hist.flatten()


def d_color_moment(im):
    R = im[:,:,0]
    G = im[:,:,1]
    B = im[:,:,2]
    rs = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    return [np.mean(R), np.std(R), np.mean(G), np.std(G), np.mean(B), np.std(B), np.mean(rs), np.std(rs)]

def imdev(im):
    [n, c, s] = np.shape(im)

    x = int(n/3)
    y = int(c/3)
    em1 = im[0:x, 0:y, :]
    em2 = im[0:x, y:y*2, :]
    em3 = im[0:x, y*2:c, :]

    em4 = im[x:2*x, 0:y, :]
    em5 = im[x:2*x, y:2*y, :]
    em6 = im[x:2*x, 2*y:c, :]

    em7 = im[2*x:n, 0:y, :]
    em8 = im[2*x:n, y:2*y, :]
    em9 = im[2*x:n, 2*y:c, :]
    
    return [em1, em2, em3, em4, em5, em6, em7, em8, em9]

def d_im9feature(im):
    em = imdev(im)
    res = []
    for e in em:
        mo = d_hu_moments(e)
        ha = d_haralick(e)
        hi = d_histogram(e)
        co = d_color_moment(e)
        res.append(np.hstack((mo, ha, hi, co)))

    return [i for x in res for i in x]


def show9im(em):
    
    f, ax = plt.subplots(3,3)
    [axi.set_axis_off() for axi in ax.ravel()]
    ax[0,0].imshow(em[0])
    ax[0,1].imshow(em[1])
    ax[0,2].imshow(em[2])
    ax[1,0].imshow(em[3])
    ax[1,1].imshow(em[4])
    ax[1,2].imshow(em[5])
    ax[2,0].imshow(em[6])
    ax[2,1].imshow(em[7])
    ax[2,2].imshow(em[8])

    plt.show()


def d_find_contours(image):
        return cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

