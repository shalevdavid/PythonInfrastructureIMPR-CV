import numpy as np
import cv2
import matplotlib.pyplot as plt

def retrunRandomMatrixWithMinMax(N):
    res = [];
    res.append(np.asmatrix(np.random.random((N,N))))

    minVal = res[0].min();
    minIndexRow = int(res[0].argmin() / N);
    minIndexCol = int(res[0].argmin() % N);

    res.append(minVal)
    res.append(minIndexRow)
    res.append(minIndexCol)

    maxVal = res[0].max();
    maxIndexRow = int(res[0].argmax() / N);
    maxIndexCol = int(res[0].argmax() % N);

    res.append(maxVal)
    res.append(maxIndexRow)
    res.append(maxIndexCol)

    return res

def cartesian2polar2D(coords):
    x = coords[:,0]
    y = coords[:,1]
    p_coords = np.zeros_like(coords)
    p_coords[:,0] = np.sqrt(x**2+y**2)
    p_coords[:,1] = np.arctan2(y,x)

    return p_coords

def convertRGB2Gray(img, type = None):

    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]

    if (type=='Lightness'):
        res = (np.minimum(R,G,B) + np.maximum(R,G,B))/2;
    else:
        if (type=='Average'):
            res = (R+G+B)/3;
        else:
            res = 0.21*R + 0.72*G + 0.07*G;

    return res

