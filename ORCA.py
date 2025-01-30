import os, glob, shutil
import numpy as np, tensorflow as tf
from tensorflow_addons.layers import GroupNormalization
from scipy import ndimage
import openslide, czifile
from jarvis.utils import arrays as jars
from jarvis.utils.general import tools as jtools
from jarvis.utils.general import gpus
from jarvis.tools import show
from jarvis.utils.display import montage
import matplotlib.pyplot as plt

# =================================================================================
# BLOBS LIBRARY
# =================================================================================

def find_bounds(msk, axes=[True, True, True], padding=[0, 0, 0], aspect_ratio=[None, None, None], dims=[1, 1, 1]):
    """
    Method to find bounds of provided mask 

    As needed, recommend using find_largest(msk, n) to preprocess mask prior to this method.

    :return

      (np.array) bounds = [

          [z0, y0, x0],
          [z1, y1, x1]]

    """
    assert msk.ndim == 4

    bounds = []

    # --- Prepare bounds
    for axis, crop in enumerate(axes):
        if crop:
            a = [0, 1, 2, 3]
            a.pop(axis)
            a = np.nonzero(np.sum(msk, axis=tuple(a)))[0]
            bounds.append([a[0], a[-1]])
        else:
            bounds.append([0, msk.shape[axis] - 1])

    # --- Prepare padding
    padding = calculate_padding(padding, np.diff(bounds, axis=1)) 
    bounds = np.array(bounds) + padding

    # --- Prepare aspect ratio
    if any([a is not None for a in aspect_ratio]):
        bounds = balance_aspect_ratio(bounds, aspect_ratio, dims)

    return bounds.T

def calculate_padding(padding, shape):
    """
    Method to calculate padding:
    
      (1) padding < 1: padding as a function (%) of the shape of blob
      (2) padding > 1: padding as absolute value 

    """
    padding = [int(np.ceil(p * s)) if p < 1 else p for p, s in zip(padding, shape[:3])] 
    padding = np.array([(p * -1, p) for p in padding])

    return padding

def balance_aspect_ratio(bounds, aspect_ratio, dims):
    """
    Method to balance aspect ratio by expanding bounds as needed along each axes

    """
    # --- Find midpoints 
    midpts = np.mean(bounds, axis=1).ravel() 

    # --- Find shapes 
    shapes = np.diff(bounds.astype('float32'), axis=1).ravel()
    shapes_zyx = shapes * np.array(dims[:3])
    shapes_norm = shapes_zyx / min([s for s in shapes_zyx if s != 0])

    none_div = lambda x, y : x / y if x is not None else None
    none_min = lambda it : min([i for i in it if i is not None])

    # --- Find ratios
    min_rs = none_min(aspect_ratio) 
    ratios = [none_div(r, min_rs * s) for r, s in zip(aspect_ratio, shapes_norm)]

    # --- Normalize 
    min_rs = none_min(ratios) 
    ratios = [r / min_rs if r is not None else 1 for r in ratios]

    shapes *= np.array(ratios)

    return np.array([midpts - shapes / 2, midpts + shapes / 2]).T

def find_largest(msk, n=1, min_ratio=None, return_labels=False):
    """
    Method to return largest n blob(s) 
    
    :params

      (int)   n            : n-largest blobs to return
      (float) min_ratio    : return blobs > min_ratio of largest blob
      (bool) return_labels : if True, return labeled matrix (instead of binary)

    """
    if not msk.any():
        return

    labels, _ = ndimage.label(msk > 0)

    counts = np.bincount(labels.ravel())[1:]
    argsrt = np.argsort(counts)[::-1] + 1

    if n == 1 and min_ratio is None:

        if return_labels:
            return labels == argsrt[0], 1

        else:
            return labels == argsrt[0]

    elif min_ratio == 0:

        if return_labels:
            return labels, counts.size

        else:
            return labels > 0


    else:

        # --- Set n based on min_ratio
        if min_ratio is not None:
            counts = counts[argsrt - 1]
            n = np.count_nonzero(counts > (counts[0] * min_ratio))

        # --- Find largest
        if return_labels:

            msk_ = np.zeros(labels.shape, dtype='int16')
            for i, a in enumerate(argsrt[:n]):
                msk_ += (labels == a) * (i + 1)

            return msk_, i + 1

        else:

            msk_ = np.zeros(labels.shape, dtype='bool')
            for a in argsrt[:n]:
                msk_ = msk_ | (labels == a)

            return msk_

def label_largest(msk, n=1, min_ratio=None):
    """
    Method to find larget n blob(s) and return labeled array

    :return

      (np.array) labels where,
        
        labels == 1 : largest blob
        labels == 2 : second largest blob
        labels == 3 : third largest blob, ...

    """
    if not msk.any():
        return

    labels, _ = ndimage.label(msk > 0)

    counts = np.bincount(labels.ravel())[1:]
    argsrt = np.argsort(counts)[::-1] + 1

    if n == 1 and min_ratio is None:
        return (labels == argsrt[0]).astype('int32')

    else:
        if min_ratio is not None:
            counts = counts[argsrt - 1]
            n = np.count_nonzero(counts >= (counts[0] * min_ratio))

        msk_ = np.zeros(labels.shape, dtype='int32')
        for i, a in enumerate(argsrt[:n]):
            msk_[labels == a] = i + 1

        return msk_, n

def perim(msk, radius=1):
    """
    Method to create msk perimeter

    """
    return ndimage.binary_dilation(msk, iterations=radius) ^ (msk > 0)

def find_center_of_mass(msk, acceleration=1):

    msk_ = msk[..., 0] if msk.ndim == 4 else msk

    # --- Subsample
    if type(acceleration) is int:
        acceleration = [acceleration] * msk_.ndim 

    msk_ = msk_[::acceleration[0], ::acceleration[1], ::acceleration[2]]
    acceleration = np.array(acceleration).reshape(1, -1)

    center = ndimage.center_of_mass(msk_)

    return np.array(center) * acceleration

def imfill(msk):
    """
    Method to 2D fill holes in blob

    """
    assert msk.ndim == 4

    filled = np.zeros(msk.shape[:3], dtype='bool')
    
    # --- Create edge mask
    edge_mask = np.zeros(shape=msk.shape[1:3], dtype='bool')
    edge_mask[0, :] = True
    edge_mask[:, 0] = True
    edge_mask[-1:, :] = True
    edge_mask[:, -1:] = True

    # --- Loop
    for z, m in enumerate(msk[..., 0]):
        if m.any():

            labels, _ = ndimage.label(m == 0)
            edges = np.unique(labels[edge_mask & (m == 0)])
            for edge in edges:
                filled[z] = filled[z] | (labels == edge)

        else:
            filled[z] = True

    return ~np.expand_dims(filled, axis=-1)

def areaopen(msk, n):
    """
    Method to return only blobs > n pixels in size

    :params

      N : size threshold (in pixels or % of maximum blob size)

    """
    if not msk.any():
        return msk

    labels = ndimage.label(msk > 0)[0]
    counts = np.bincount(labels.ravel())[1:]

    if n < 1:
        n = np.max(counts) * n

    msk_ = np.zeros(msk.shape)
    inds = np.nonzero(counts >= n)[0] + 1

    for ind in inds:
        msk_[labels == ind] = 1

    return msk_ 

if __name__ == '__main__':

    pass

###### END BLOBS LIBRARY ######

CODE = jtools.get_paths('ws/arterio')['code']

def load_region(f, y, x, h=512, w=512, mask=None, **kwargs):

    if mask is not None:
        x0, y0, x1, y1 = load_coords_from_mask(y, x, mask, **kwargs)

    else:
        x0 = x
        y0 = y
        x1 = x + w 
        y1 = y + h

    if hasattr(f, 'read_region'):
        return load_region_svs(f, y0, x0, y1, x1), y0, x0

    else:
        return load_region_czi(f, y0, x0, y1, x1), y0, x0

def load_coords_from_mask(y, x, mask, padding=(0, 0.5, 0.5), min_shape=512, **kwargs):

    if mask.ndim == 3:
        mask = np.expand_dims(mask, axis=0)

    # --- Find largest blob
    mask = find_largest(mask, n=1)

    # --- Find bounds
    bounds = find_bounds(mask, padding=padding, aspect_ratio=(None, 1, 1))

    # --- Ensure minimum shape
    if np.diff(bounds[:, 1]) < min_shape:
        center = np.mean(bounds, axis=0)
        bounds = np.stack((
            center - (min_shape / 2),
            center + (min_shape / 2)), axis=0)
        bounds[:, 0] = 0

    bounds = np.round(bounds).astype('int')
    y0, y1 = bounds[:, 1]
    x0, x1 = bounds[:, 2]

    return x + x0, y + y0, x + x1, y + y1

def load_region_svs(f, y0, x0, y1, x1):

    return np.array(f.read_region((x0, y0), 0, (x1 - x0, y1 - y0)))

def load_region_czi(f, y0, x0, y1, x1):

    pad_x = (0, 0)
    pad_y = (0, 0)

    if y0 < 0:
        pass

    sub = f[..., y0:y1, x0:x1, :]

    if any(pad_x) or any(pad_y):
        pass

    return sub

def load_raw(fname):

    if fname[-3:] == 'svs':

        f = openslide.OpenSlide(fname)
        x, y = f.dimensions

        try:
            d = float(f.properties['tiff.ImageDescription'].split('MPP = ')[1].split('|')[0]) 
        except:
            d = 0.2738

        return f, x, y, d
    
    # --- chanon edited
    if fname[-3:] == 'czi':
        
        f = np.squeeze(czifile.imread(fname))
        y = f.shape[0]
        x = f.shape[1]
        d = 0.2738

        return f, x, y, d

    assert False, 'ERROR fname ext not supported: {}'.format(fname) 
    
def predict(model, dat):

    dat = resample(dat)
    nrm = (dat - 128) / 255

    lbl = np.squeeze(model.predict(nrm.reshape(1, 1, 256, 256, 3))['lbl'])
    sfx = np.exp(lbl) / np.sum(np.exp(lbl), axis=-1, keepdims=True)
    sfx = ndimage.gaussian_filter(sfx[..., 1:2], sigma=6)

    return sfx, dat

def resample(dat, shape=(256, 256, 3)):

    dat = dat[..., :3]

    zoom = np.array(shape) / np.array(dat.shape)

    return ndimage.zoom(dat, zoom, order=1)

def create_predictions_recursive(model, f, y0, x0, sfx_old=None, sfx_thresh=0.78, sfx_count=1900, count=0, count_max=10, tolerance=1e-2, bg_thresh=200, **kwargs):

    # --- Load data
    mask = None if sfx_old is None else sfx_old > sfx_thresh

    dat, y0, x0 = load_region(f, y=y0, x=x0, mask=mask, **kwargs)

    # --- Check background mean
    if dat[..., :3][dat[..., :3] > 0].mean() > bg_thresh:
        return dat, None

    # --- Predict
    sfx_new, dat_ = predict(model, dat)
    sfx_new_ = find_largest(np.expand_dims(sfx_new, axis=0) > sfx_thresh)
    sfx_new_ = sfx_new_[0] if sfx_new_ is not None else sfx_new

    # --- Check cutoff
    if np.count_nonzero(sfx_new_ > sfx_thresh) < sfx_count:
        return dat, None 

    # --- Check prediction mean
    m = np.concatenate([sfx_new_] * 3, axis=-1)
    if dat_[..., :3][m > sfx_thresh].mean() > bg_thresh:
        return dat, None

    # --- Check counts
    count += 1
    if count > count_max:
        return dat, sfx_new

    # --- Check for center predictions
    if count > 0:
        if not sfx_new_[128-16:128+16, 128-16:128+16].any():
            return dat, None

    if sfx_old is not None:

        # --- Check tolerance 
        if np.abs(sfx_old.mean() - sfx_new.mean()) < tolerance:
            return dat, sfx_new

        # --- Check shape change
        if sfx_old.shape[0] > dat.shape[0]:
            return dat, sfx_new

    # --- Run iteratively
    sfx_old = resample(sfx_new, shape=dat[..., :1].shape)

    return create_predictions_recursive(model, f, y0, x0, sfx_old=sfx_old, count=count)

def create_weight_standardization(kernel):
    """
    Method to apply weight standardization to 2D kernel

    """
    axis = (0, 1, 2) if len(kernel.shape) == 4 else (0, 1, 2, 3)

    mu = tf.math.reduce_mean(kernel, axis=axis, keepdims=True)
    sd = tf.math.reduce_std(kernel, axis=axis, keepdims=True)

    kernel = tf.math.divide_no_nan(kernel - mu, sd)

def create_predictions(fname, model='{}/exps/augs/exp-01-0/hdf5/model_040.hdf5', sfx_thresh=0.78, cnt_thresh=1900, draw_hmap=True, **kwargs):

    # --- Load raw
    f, X, Y, D = load_raw(fname)

    # --- Load model
    model = tf.keras.models.load_model(model.format(CODE), compile=False, custom_objects={
        'create_weight_standardization': create_weight_standardization})

    # --- Loop
    for y0 in range(0, Y, 512):
        for x0 in range(0, X, 512):

            print('Running: x = {:07d}/{:07d} | y = {:07d}/{:07d}'.format(x0, X, y0, Y), end='\r', flush=True)
            try:
                dat, sfx = create_predictions_recursive(model, f, y0, x0)
            except:
                sfx = None
                print('WARNING: Create predictions recursive failed.' )

            if sfx is not None:

                rat = (dat.shape[0] / 512) * (dat.shape[1] / 512) * D

                output = '{}/data/pngs/{}/{}/{:07d}-sd-{:05d}-y-{:07d}-x-{:07d}.png'.format(
                    CODE,
                    fname.split('/')[-2],
                    fname.split('/')[-1][:-4],
                    int(np.count_nonzero(sfx > sfx_thresh) * rat),
                    int(np.std(sfx) * 10000),
                    y0,
                    x0)
                
                os.makedirs(os.path.dirname(output), exist_ok=True)

                if draw_hmap:
                    arr = draw_overlay_summary(resample(dat), sfx[..., 0])
                    plt.imsave(output, arr)

                else:
                    # ========================================
                    # ADD POTENTIAL RESAMPLE
                    # ========================================
                    plt.imsave(output, dat)

def draw_overlay_summary(dat, sfx, **kwargs):

    # --- Create heatmap
    hm = plt.get_cmap('plasma')(sfx) * 255
    hm = np.round(hm[..., :3]).astype('uint8')

    # --- Create concatenated
    arr = np.stack((dat, hm))
    arr = montage(arr, rgb=True)

    return arr[0, :dat.shape[0]]

def filter_pngs(sd=2400, **kwargs):

    for subdir in sorted(glob.glob('./pngs/*/*/')):

        pngs = sorted(glob.glob('{}*.png'.format(subdir)))
        os.makedirs(subdir + 'filtered/', exist_ok=True)

        for n, png in enumerate(pngs):

            print('Filtering ({:06d}): {}'.format(n, subdir))

            if int(png.split('-sd-')[1].split('-y')[0]) < sd:
                dst = png.replace(subdir, subdir + 'filtered/')
                shutil.move(src=png, dst=dst)

if __name__ == '__main__':

    # --- Autoselect GPU
    gpus.autoselect()

    fnames = [
        '/data/raw/wsi_arterio/data/V019_B1_HE/V019_B1_HE.svs',
	'/data/raw/wsi_arterio/data/V019_B3_HE/V019_B3_HE.svs'
        ]

    # --- Create predictions
    for fname in fnames:
        create_predictions(fname=fname, draw_hmap=False)

    # --- Filter predictions
    filter_pngs()
