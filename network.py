from tensorflow.keras.layers import *
from tensorflow.keras.models import *

def unet(n_class, use_bias=True, mode='2d'):
    
    _input = {
        '2d':(None, None, 1),
        '3d':(None, None, None, 1)
    }
    conv = {
        '2d': Conv2D,
        '3d': Conv3D
    }
    
    upconv = {
        '2d': Conv2DTranspose,
        '3d': Conv3DTranspose
    }
    pool = {
        '2d': MaxPool2D,
        '3d': MaxPool3D
    }
    
    _input = Input(shape=_input[mode])
#     _input = Input(shape=(None, None, 1))
    en1 = conv[mode](64, 3, strides=1, padding='same', activation='relu', use_bias=use_bias)(_input)
    en1 = conv[mode](64, 3, strides=1, padding='same', activation='relu', use_bias=use_bias)(en1)
    
    en2 = pool[mode]()(en1)
    en2 = conv[mode](128, 3, strides=1, padding='same', activation='relu', use_bias=use_bias)(en2)
    en2 = conv[mode](128, 3, strides=1, padding='same', activation='relu', use_bias=use_bias)(en2)
    
    en3 = pool[mode]()(en2)
    en3 = conv[mode](256, 3, strides=1, padding='same', activation='relu', use_bias=use_bias)(en3)
    en3 = conv[mode](256, 3, strides=1, padding='same', activation='relu', use_bias=use_bias)(en3)
    
    en4 = pool[mode]()(en3)
    en4 = conv[mode](512, 3, strides=1, padding='same', activation='relu', use_bias=use_bias)(en4)
    en4 = conv[mode](512, 3, strides=1, padding='same', activation='relu', use_bias=use_bias)(en4)
    
    en5 = pool[mode]()(en4)
    en5 = conv[mode](1024, 3, strides=1, padding='same', activation='relu', use_bias=use_bias)(en5)
    en5 = conv[mode](1024, 3, strides=1, padding='same', activation='relu', use_bias=use_bias)(en5)
    
    de4 = upconv[mode](512, 2, strides=2, padding='same', activation='relu', use_bias=use_bias)(en5)
    de4 = Concatenate(axis=-1)([en4, de4])
    de4 = conv[mode](512, 3, strides=1, padding='same', activation='relu', use_bias=use_bias)(de4)
    de4 = conv[mode](512, 3, strides=1, padding='same', activation='relu', use_bias=use_bias)(de4)
    
    de3 = upconv[mode](256, 2, strides=2, padding='same', activation='relu', use_bias=use_bias)(de4)
    de3 = Concatenate(axis=-1)([en3, de3])
    de3 = conv[mode](256, 3, strides=1, padding='same', activation='relu', use_bias=use_bias)(de3)
    de3 = conv[mode](256, 3, strides=1, padding='same', activation='relu', use_bias=use_bias)(de3)
    
    de2 = upconv[mode](128, 2, strides=2, padding='same', activation='relu', use_bias=use_bias)(de3)
    de2 = Concatenate(axis=-1)([en2, de2])
    de2 = conv[mode](128, 3, strides=1, padding='same', activation='relu', use_bias=use_bias)(de2)
    de2 = conv[mode](128, 3, strides=1, padding='same', activation='relu', use_bias=use_bias)(de2)
    
    de1 = upconv[mode](64, 2, strides=2, padding='same', activation='relu', use_bias=use_bias)(de2)
    de1 = Concatenate(axis=-1)([en1, de1])
    de1 = conv[mode](64, 3, strides=1, padding='same', activation='relu', use_bias=use_bias)(de1)
    de1 = conv[mode](64, 3, strides=1, padding='same', activation='relu', use_bias=use_bias)(de1)
    
    pred = conv[mode](n_class, 1, strides=1, padding='same', activation='softmax', use_bias=use_bias)(de1)
    return Model(inputs = _input, outputs=pred)