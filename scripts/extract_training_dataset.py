import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # hide info and warning messages
import warnings
warnings.filterwarnings('ignore') #filter warning messages
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) #only show tf errors
from galaxy2galaxy import problems
from absl import flags, app
import numpy as np
import galflow as gf
from scipy import fft
from absl import app

flags.DEFINE_string(
    'data_dir',
    default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'), 'data/'),
    help="Directory to read the data from.")

FLAGS = flags.FLAGS

#This script will extract the generated images from galaxy2galaxy, allowing us to use the same
#dataset in different models.

#to run, just pass in the data_directory of the generated files from galaxy2galaxy
#example: --data_dir=/home/data_dir_optical/

def export_dataset(dataset, output_path):
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    elems = []
    with tf.Session() as sess:
        try:
            while True:
                elem = sess.run(next_element)
                elems.append(elem)
        except tf.errors.OutOfRangeError:
            pass

    # Check element type: tuple/list or dict
    if isinstance(elems[0], dict):
        # Save each dict key as an array
        arrays = {key: np.array([e[key] for e in elems]) for key in elems[0].keys()}
    else:
        # Assume tuple or list, save each position as separate array
        arrays = {}
        for i in range(len(elems[0])):
            arrays[f"elem_{i}"] = np.array([e[i] for e in elems])

    np.savez(output_path, **arrays)
    print(f"Saved dataset with {len(elems)} elements to {output_path}")

def main(argv):
    #tf.compat.v1.enable_eager_execution()
    Modes = tf.estimator.ModeKeys
    print("Generating dataset using galaxy2galaxy...")
    problem128 = problems.problem('attrs2img_cosmos_cfht2hst')
    dset = problem128.dataset(Modes.TRAIN, data_dir=FLAGS.data_dir)
    print("Exporting to file saved_dataset.npz...")
    export_dataset(dset, "saved_dataset.npz")
    print("All done!")


if __name__ == '__main__':
    app.run(main)
