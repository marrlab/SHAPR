from keras import backend as K
from tensorflow.keras.losses import MSE, binary_crossentropy
from shapr.utils import *

def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    """
    Adding the dice loss and the MSE loss together
    """
    return 1-dice_coef(y_true, y_pred)

def dice_crossentropy_loss(y_true, y_pred):
    """
    Adding the dice loss and the binary crossentropy as well as a penalty for the volume
    """
    return dice_coef_loss(y_true, y_pred) + binary_crossentropy(y_true, y_pred)

def mse(y_true, y_pred):
    return MSE(y_true, y_pred)

def IoU(y_true,y_pred):
    intersection = y_true + y_pred
    intersection = np.count_nonzero(intersection > 1.5)
    union = y_true + y_pred
    union = np.count_nonzero(union > 0.5)
    return intersection / union