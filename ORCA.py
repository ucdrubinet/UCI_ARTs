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
    mask = jars.blobs.find_largest(mask, n=1)

    # --- Find bounds
    bounds = jars.blobs.find_bounds(mask, padding=padding, aspect_ratio=(None, 1, 1))

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
    sfx_new_ = jars.blobs.find_largest(np.expand_dims(sfx_new, axis=0) > sfx_thresh)
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
