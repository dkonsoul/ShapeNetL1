import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) #only show tf errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # hide info and warning messagesi
import warnings
warnings.filterwarnings('ignore') #filter warning messages
import sys
sys.path.append("../ShapeNetL1")
import numpy as np
import cadmos_lib as cl
import galflow as gf
from scipy import fft
from absl import app
import matplotlib.pyplot as plt
#from PIL import Image
import tifffile

def ir2tf(imp_resp, shape):
    dim = 2
    # Zero padding and fill
    irpadded = np.zeros(shape)
    irpadded[tuple([slice(0, s) for s in imp_resp.shape])] = imp_resp
    # Roll for zero convention of the fft to avoid the phase
    # problem. Work with odd and even size.
    for axis, axis_size in enumerate(imp_resp.shape):

        irpadded = np.roll(irpadded,
                           shift=-int(np.floor(axis_size / 2)),
                           axis=axis)

    return fft.rfftn(irpadded, axes=range(-dim, 0))

def laplacian(shape):
    impr = np.zeros([3,3])
    for dim in range(2):
        idx = tuple([slice(1, 2)] * dim +
                    [slice(None)] +
                    [slice(1, 2)] * (1 - dim))
        impr[idx] = np.array([-1.0,
                              0.0,
                              -1.0]).reshape([-1 if i == dim else 1
                                              for i in range(2)])
    impr[(slice(1, 2), ) * 2] = 4.0
    return ir2tf(impr, shape), impr

def laplacian_tf(shape):
    return tf.convert_to_tensor(laplacian(shape)[0])

def wiener_tf(image, psf, balance, laplacian=True):
    r"""Applies Wiener filter to image.

    This function takes an image in the direct space and its corresponding PSF in the
    Fourier space and performs a deconvolution using the Wiener Filter.

    Parameters
    ----------
    image   : 2D TensorFlow tensor
        Image in the direct space.
    psf     : 2D TensorFlow tensor
        PSF in the Fourier space (or K space).
    balance : scalar
        Weight applied to regularization.
    laplacian : boolean
        If true the Laplacian regularization is used else the identity regularization 
        is used.

    Returns
    -------
    tuple
        The first element is the filtered image in the Fourier space.
        The second element is the PSF in the Fourier space (also know as the Transfer
        Function).
    """
    trans_func = psf
    if laplacian:
        reg = laplacian_tf(image.shape)
        if psf.shape != reg.shape:
            trans_func = tf.signal.rfft2d(tf.signal.ifftshift(tf.cast(psf, 'float32')))
        else:
            trans_func = psf
    
    arg1 = tf.cast(tf.math.conj(trans_func), 'complex64')
    arg2 = tf.dtypes.cast(tf.math.abs(trans_func),'complex64') ** 2
    arg3 = balance
    if laplacian:
        arg3 *= tf.dtypes.cast(tf.math.abs(laplacian_tf(image.shape)), 'complex64')**2
    wiener_filter = arg1 / (arg2 + arg3)
    
    # Apply wiener in Foutier (or K) space
    wiener_applied = wiener_filter * tf.signal.rfft2d(tf.cast(image, 'float32'))
    
    return wiener_applied, trans_func

def pre_proc_unet(dico):
    r"""Preprocess the data and apply the Tikhonov filter on the input galaxy images.

    This function takes the dictionnary of galaxy images and PSF for the input and
    the target and returns a list containing 2 arrays: an array of galaxy images that
    are the output of the Tikhonov filter and an array of target galaxy images.

    Parameters
    ----------
    dico : dictionnary
        Array_like means all those objects -- lists, nested lists, etc. --
        that can be converted to an array.  We can also refer to
        variables like `var1`.

    Returns
    -------
    list
        list containing 2 arrays: an array of galaxy images that are the output of the
        Tikhonov filter and an array of target galaxy images.

    Example
    -------
    These are written in doctest format, and should illustrate how to
    use the function.

    >>> from galaxy2galaxy import problems # to list avaible problems run problems.available()
    >>> problem128 = problems.problem('attrs2img_cosmos_hst2euclide')
    >>> dset = problem128.dataset(Modes.TRAIN, data_dir='attrs2img_cosmos_hst2euclide')
    >>> dset = dset.map(pre_proc_unet)
    """
    # First, we add noise
    # https://github.com/CosmoStat/ShapeDeconv/blob/master/data/CFHT/HST2CFHT.ipynb
    sigma_cfht = 23.61
    noise = tf.random_normal(shape=tf.shape(dico['inputs']), mean=0.0, stddev=sigma_cfht, dtype=tf.float32)
    dico['inputs'] = dico['inputs'] + noise
    # Normalize images to improve the training
    norm_factor = 4e3
    dico['inputs'] = dico['inputs']/norm_factor
    dico['targets'] = dico['targets']/norm_factor

    # Second, we interpolate the image on a finer grid
    x_interpolant=tf.image.ResizeMethod.BICUBIC
    interp_factor = 2
    Nx = 64
    Ny = 64
    dico['inputs_cfht'] = tf.image.resize(dico['inputs'],
                    [Nx*interp_factor,
                    Ny*interp_factor],
                    method=x_interpolant)
    # Since we lower the resolution of the image, we also scale the flux
    # accordingly
    dico['inputs_cfht'] = dico['inputs_cfht'] / interp_factor**2

    balance = 9e-3  # determined using line search
    dico['inputs_tikho'], _ = wiener_tf(dico['inputs_cfht'][...,0], dico['psf_cfht'][...,0], balance)
    dico['inputs_tikho'] = tf.expand_dims(dico['inputs_tikho'], axis=0)
    psf_hst = tf.reshape(dico['psf_hst'], [dico['psf_hst'].shape[-1],*dico['psf_hst'].shape[:2]])
    psf_hst = tf.cast(psf_hst, 'complex64')
    # gf.kconvolve performs a convolution in the K (Fourier) space
    # inputs are given in K space
    # the output is in the direct space
    dico['inputs_tikho'] = gf.kconvolve(dico['inputs_tikho'], psf_hst,zero_padding_factor=1,interp_factor=interp_factor)
    dico['inputs_tikho'] = dico['inputs_tikho'][0,...]

    return dico['inputs_tikho'], dico['targets']

def plotImagesAfter(dset, num, saveToFile=False, imgPrefix="image"):
    #Used to debug and check the dataset
    for i, element in enumerate(dset.take(num)):
        image1, image2 = element
        plt.figure(figsize=(12, 6))
        
        # Plot the first image with a color bar
        plt.subplot(1, 2, 1)
        im1 = plt.imshow(tf.squeeze(image1), cmap='gray')
        plt.title('Inputs')
        plt.axis('off')
        cbar1 = plt.colorbar(im1, fraction=0.046, pad=0.04)
        cbar1.set_ticks([tf.reduce_min(image1).numpy(), tf.reduce_max(image1).numpy()])
        
        # Plot the second image with a color bar
        plt.subplot(1, 2, 2)
        im2 = plt.imshow(tf.squeeze(image2), cmap='gray')
        plt.title('Targets')
        plt.axis('off')
        cbar2 = plt.colorbar(im2, fraction=0.046, pad=0.04)
        cbar2.set_ticks([tf.reduce_min(image2).numpy(), tf.reduce_max(image2).numpy()])
        
        # Show the plots
        plt.tight_layout()
        if saveToFile:
            save_path = f"{imgPrefix}_{i}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Create subfolders if they don't exist
            plt.savefig(save_path)
        else:
            plt.show()

def export_images(dset, limit=80000, base_dir="dataset"):
    inputs_dir_train = os.path.join(base_dir, "train/input")
    inputs_dir_test = os.path.join(base_dir, "test/input")
    targets_dir_train = os.path.join(base_dir, "train/target")
    targets_dir_test = os.path.join(base_dir, "test/target")
    validation_perc = 10 #Use this percentage to determine what percentage of the images will be used for model's validation
    os.makedirs(inputs_dir_train, exist_ok=True)
    os.makedirs(inputs_dir_test, exist_ok=True)
    os.makedirs(targets_dir_train, exist_ok=True)
    os.makedirs(targets_dir_test, exist_ok=True)

    for i, element in enumerate(dset.take(limit)):
        if (i+1)%1000==0:
            print("Progress:",i+1,"/",limit)
        
        image1, image2 = element

        # Remove batch dimension and convert tensor to numpy array
        img1_np = tf.squeeze(image1).numpy()
        img2_np = tf.squeeze(image2).numpy()

        if(i<limit-limit*validation_perc/100):
            # Save input image
            input_path = os.path.join(inputs_dir_train, f"{1+i}.tiff")
            tifffile.imwrite(input_path, img1_np, dtype=img1_np.dtype)
            
            # Save target image
            target_path = os.path.join(targets_dir_train, f"{1+i}.tiff")
            tifffile.imwrite(target_path, img2_np, dtype=img2_np.dtype)
        else: #handle the test data (get some percentage for test)
            input_path = os.path.join(inputs_dir_test, f"{1+i-int(limit-limit*validation_perc/100)}.tiff")
            tifffile.imwrite(input_path, img1_np, dtype=img1_np.dtype)
            target_path = os.path.join(targets_dir_test, f"{1+i-int(limit-limit*validation_perc/100)}.tiff")
            tifffile.imwrite(target_path, img2_np, dtype=img2_np.dtype)

def load_dataset(input_path):
    data = np.load(input_path)
    if all(k.startswith('elem_') for k in data.files):
        elems = tuple(data[k] for k in sorted(data.files))
        dataset = tf.data.Dataset.from_tensor_slices(elems)
    else:
        dict_slices = {k: data[k] for k in data.files}
        dataset = tf.data.Dataset.from_tensor_slices(dict_slices)
    return dataset


def main(argv):
    tf.compat.v1.enable_eager_execution()
    print("Loading the dataset from saved_dataset.npz...")
    dset = load_dataset("saved_dataset.npz")
    print("Dataset loaded! Pre-proccessing images...")
    dset = dset.repeat()
    dset = dset.map(pre_proc_unet)
    print("Done! Exporting images...")
    #plotImagesAfter(dset, 1, False, "")
    export_images(dset)
    print("Process complete!")

if __name__ == '__main__':
    app.run(main)
