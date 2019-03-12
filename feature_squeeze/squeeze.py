import torch
import numpy as np
from scipy import ndimage
import torch.distributions as tdist
import copy

def recreate_image(x, npp_int):
    """
        Recreates images from a torch Tensor, sort of reverse preprocessing

    Args:
        x (np.array): C,H,W format Image to recreate

    returns:
        recreated_im (numpy arr): H,W,C format Recreated image in array
    """
    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1/0.229, 1/0.224, 1/0.225]
    in_channel = x.shape[0]
    recreated_im = copy.copy(x)  # C, H, W
    if in_channel == 3:
        for c in range(in_channel):
            recreated_im[c] /= reverse_std[c]
            recreated_im[c] -= reverse_mean[c]
    elif in_channel == 1:
        recreated_im[0] /= reverse_std[1]
        recreated_im[0] -= reverse_mean[1]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.rint(recreated_im * npp_int)
    recreated_im = np.transpose(recreated_im, axes=(1, 2, 0))  # H, W, C
    return recreated_im

def normalized_process(x, npp_int):
    """
    :param x: H,W,C format
    :return:  C,H,W format
    """
    in_channel = x.shape[-1]
    if in_channel == 3:
        mean = np.expand_dims(np.expand_dims(np.array([0.485, 0.456, 0.406]),1),1)  # 3,1,1
        std = np.expand_dims(np.expand_dims(np.array([0.229, 0.224, 0.225]),1),1)   # 3,1,1
    elif in_channel == 1:
        mean = np.expand_dims(np.expand_dims(np.array([0.456]),1),1)  # 1,1,1
        std = np.expand_dims(np.expand_dims(np.array([0.224]), 1), 1)  # 1,1,1
    processed_x = copy.copy(x)  # H,W,C
    processed_x = np.transpose(processed_x,(2,0,1)) # C,H,W
    processed_x = processed_x.astype(np.float32)
    processed_x = processed_x / npp_int  # normalize
    processed_x = (processed_x - mean) / std
    return processed_x

def reduce_precision_py(x, npp):
    """
    Reduce the precision of image, the numpy version.
    :param x: a float tensor, which has been scaled to [0, 1].
    :param npp: number of possible values per pixel. E.g. it's 256 for 8-bit gray-scale image, and 2 for binarized image.
    :return: a tensor representing image(s) with lower precision.
    """
    # Note: 0 is a possible value too.
    npp_int = npp - 1
    assert x.ndim == 4, x.ndim
    new_x = []
    for x_ in x:
        x_int = recreate_image(x_, npp_int)  # H,W,C
        x_float = normalized_process(x_int, npp_int)  # C,H,W
        new_x.append(x_float)
    return np.stack(new_x)


def bit_depth_py(x, bits):
    precisions = 2**bits
    return reduce_precision_py(x, precisions)


def bit_depth_random_py(x, bits, stddev):
    if stddev == 0.:
        rand_array = np.zeros(x.shape)
    else:
        rand_array = np.random.normal(loc=0., scale=stddev, size=x.shape)
    x_random = np.add(x, rand_array)
    return bit_depth_py(x_random, bits)

def binary_filter_py(x, threshold):
    """
    An efficient implementation of reduce_precision_py(x, 2). 灰度二值化
    """
    x_bin = np.maximum(np.sign(x - threshold), 0)
    return x_bin


def binary_random_filter_py(x, threshold, stddev=0.125):
    if stddev == 0.:
        rand_array = np.zeros(x.shape)
    else:
        rand_array = np.random.normal(loc=0., scale=stddev, size=x.shape)
    x_bin = np.maximum(np.sign(np.add(x, rand_array)-threshold), 0).astype(np.float32)
    return x_bin


def median_filter_py(x, width, height=-1):
    """
    Median smoothing by Scipy.
    :param x: a tensor of image(s)
    :param width: the width of the sliding window (number of pixels)
    :param height: the height of the window. The same as width by default.
    :return: a modified tensor with the same shape as x.
    """
    if height == -1:
        height = width
    return ndimage.filters.median_filter(x, size=(1, 1, width, height), mode='reflect')

# Squeezers implemented in OpenCV
# OpenCV expects uint8 as image data type.
def opencv_wrapper(imgs, opencv_func, argv):
    # imgs is N,C,H,W format
    ret_imgs = []
    imgs_copy = imgs

    for img in imgs_copy:  # each is C,H,W
        img = recreate_image(img)  # H, W, C
        if img.shape[-1] == 1:
            img = np.squeeze(img) # grey image convert to H,W
        img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
        ret_img = opencv_func(*[img_uint8]+argv)  # H,W,C
        if type(ret_img) == tuple:
            ret_img = ret_img[1]
        if ret_img.shape[-1] == 1:
            ret_img = np.expand_dims(ret_img, -1) # H, W, 1
        ret_img = normalized_process(ret_img, 255).astype(np.float32) # C,H,W
        ret_imgs.append(ret_img)
    ret_imgs = np.stack(ret_imgs) # N, C,H ,W
    return ret_imgs

def otsu_binarize_py(x):
    # func = lambda img: cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    # return opencv_binarize(x, func)
    import cv2
    ret_imgs = opencv_wrapper(x, cv2.threshold, [0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU])
    return ret_imgs


def adaptive_binarize_py(x, block_size=5, C=33.8):
    "Works like an edge detector."
    # ADAPTIVE_THRESH_GAUSSIAN_C, ADAPTIVE_THRESH_MEAN_C
    # THRESH_BINARY, THRESH_BINARY_INV
    import cv2
    ret_imgs = opencv_wrapper(x, cv2.adaptiveThreshold, [255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, C])
    return ret_imgs

# Non-local Means
def non_local_means_color_py(imgs, search_window, block_size, photo_render):
    import cv2
    ret_imgs = opencv_wrapper(imgs, cv2.fastNlMeansDenoisingColored, [None,photo_render,photo_render,block_size,search_window])
    return ret_imgs

def non_local_means_color_torch(imgs, search_window, block_size, photo_render):
    if isinstance(imgs, torch.Tensor):
        imgs = imgs.detach().cpu().numpy()
    ret = torch.from_numpy(non_local_means_color_py(imgs, search_window, block_size, photo_render))
    return ret

def non_local_means_bw_py(imgs, search_window, block_size, photo_render):
    import cv2
    ret_imgs = opencv_wrapper(imgs, cv2.fastNlMeansDenoising, [None,photo_render,block_size,search_window])
    return ret_imgs

def non_local_means_bw_torch(imgs, search_window, block_size, photo_render):
    if isinstance(imgs, torch.Tensor):
        imgs = imgs.detach().cpu().numpy()
    ret = torch.from_numpy(non_local_means_bw_py(imgs, search_window, block_size, photo_render))
    return ret

def bilateral_filter_py(imgs, d, sigmaSpace, sigmaColor):
    """
    :param d: Diameter of each pixel neighborhood that is used during filtering.
        If it is non-positive, it is computed from sigmaSpace.
    :param sigmaSpace: Filter sigma in the coordinate space.
        A larger value of the parameter means that farther pixels will influence each other as long as their colors are close enough (see sigmaColor ).
        When d>0, it specifies the neighborhood size regardless of sigmaSpace.
        Otherwise, d is proportional to sigmaSpace.
    :param sigmaColor: Filter sigma in the color space.
        A larger value of the parameter means that farther colors within the pixel neighborhood (see sigmaSpace) will be mixed together, resulting in larger areas of semi-equal color.
    """
    import cv2
    return opencv_wrapper(imgs, cv2.bilateralFilter, [d, sigmaColor, sigmaSpace])

def adaptive_bilateral_filter_py(imgs, ksize, sigmaSpace, maxSigmaColor=20.0):
    import cv2
    return opencv_wrapper(imgs, cv2.adaptiveBilateralFilter, [(ksize,ksize), sigmaSpace, maxSigmaColor])



def isfloat(value):
    try:
        float(value)
        return True
    except:
        return False

def parse_params(params_str):
    params = []
    for param in params_str.split('_'):
        param = param.strip()
        if param.isdigit():
            param = int(param)
        elif isfloat(param):
            param = float(param)
        else:
            continue
        params.append(param)
    return params

def get_squeezer_by_name(name, func_type):
    squeezer_list = ['none',
                     'bit_depth_random',
                     'bit_depth',
                     'binary_filter',
                     'reduce_precision',
                     'binary_random_filter',
                     'adaptive_binarize',
                     'otsu_binarize',
                     'median_filter',
                     'median_random_filter',
                     'median_random_size_filter',
                     'non_local_means_bw',
                     'non_local_means_color',
                     'adaptive_bilateral_filter',
                     'bilateral_filter',
                    ]

    for squeezer_name in squeezer_list:
        if name.startswith(squeezer_name):
            params_str = name[len(squeezer_name):]
            func_name = "%s_py" % squeezer_name if func_type == 'python' else "%s_tf" % squeezer_name
            # Return a list
            args = parse_params(params_str)
            # print ("params_str: %s, args: %s" % (params_str, args))
            return lambda x: globals()[func_name](*([x] + args))

    raise Exception('Unknown squeezer name: {} squeezer_name:{}'.format(name, squeezer_name))

def get_sequential_squeezers_by_name(squeezers_name):
    # example_squeezers_name = "binary_filter_0.5,median_filter_2_2"
    squeeze_func = None
    for squeezer_name in squeezers_name.split(','):
        squeezer = get_squeezer_by_name(squeezer_name, 'python')
        if squeeze_func == None:
            squeeze_func = lambda x: squeezer(x)
        else:
            old_func = squeeze_func
            squeeze_func = lambda x: squeezer(old_func(x))
    return squeeze_func



