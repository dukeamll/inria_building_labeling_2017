import os
import time
import argparse
import numpy as np
import tensorflow as tf
import utils
from network import unet
from dataReader import image_reader, patch_extractor
from rsrClassData import rsrClassData

TRAIN_DATA_DIR = 'dcc_inria_train'
VALID_DATA_DIR = 'dcc_inria_valid'
CITY_NAME = 'austin,chicago,kitsap,tyrol-w,vienna'
RSR_DATA_DIR = r'/media/ei-edl01/data/remote_sensing_data'
PATCH_DIR = r'/home/lab/Documents/bohao/data/inria'
TRAIN_PATCH_APPENDIX = 'train_noaug_dcc'
VALID_PATCH_APPENDIX = 'valid_noaug_dcc'
TRAIN_TILE_NAMES = ','.join(['{}'.format(i) for i in range(1,32)])
VALID_TILE_NAMES = ','.join(['{}'.format(i) for i in range(32,37)])
RANDOM_SEED = 1234
BATCH_SIZE = 5
LEARNING_RATE = 1e-3
INPUT_SIZE = 572
EPOCHS = 100
CKDIR = r'./models'
MODEL_NAME = 'UnetInria_fr_mean_reduced_appendix_large'
DATA_AUG = 'filp,rotate'
NUM_CLASS = 2
N_TRAIN = 8000
GPU = '1'
DECAY_STEP = 60
DECAY_RATE = 0.1
VALID_SIZE = 1000


def read_flag():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data-dir', default=TRAIN_DATA_DIR, help='path to release folder')
    parser.add_argument('--valid-data-dir', default=VALID_DATA_DIR, help='path to release folder')
    parser.add_argument('--rsr-data-dir', default=RSR_DATA_DIR, help='path to rsrClassData folder')
    parser.add_argument('--patch-dir', default=PATCH_DIR, help='path to patch directory')
    parser.add_argument('--train-patch-appendix', default=TRAIN_PATCH_APPENDIX, help='train patch appendix')
    parser.add_argument('--valid-patch-appendix', default=VALID_PATCH_APPENDIX, help='valid patch appendix')
    parser.add_argument('--train-tile-names', default=TRAIN_TILE_NAMES, help='image tiles')
    parser.add_argument('--valid-tile-names', default=VALID_TILE_NAMES, help='image tiles')
    parser.add_argument('--city-name', type=str, default=CITY_NAME, help='city name (default austin)')
    parser.add_argument('--random-seed', type=int, default=RANDOM_SEED, help='tf random seed')
    parser.add_argument('--batch-size', default=BATCH_SIZE, type=int, help='batch size (10)')
    parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE, help='learning rate (1e-3)')
    parser.add_argument('--input-size', default=INPUT_SIZE, type=int, help='input size 224')
    parser.add_argument('--epochs', default=EPOCHS, type=int, help='# epochs (1)')
    parser.add_argument('--ckdir', default=CKDIR, help='ckpt dir (models)')
    parser.add_argument('--num-classes', type=int, default=NUM_CLASS, help='# classes (including background)')
    parser.add_argument('--n-train', type=int, default=N_TRAIN, help='# samples per epoch')
    parser.add_argument('--GPU', type=str, default=GPU, help="GPU used for computation.")
    parser.add_argument('--decay-step', type=float, default=DECAY_STEP, help='Learning rate decay step in number of epochs.')
    parser.add_argument('--decay-rate', type=float, default=DECAY_RATE, help='Learning rate decay rate')
    parser.add_argument('--valid-size', type=int, default=VALID_SIZE, help='#patches to valid')
    parser.add_argument('--model-name', type=str, default=MODEL_NAME, help='Model name')
    parser.add_argument('--data-aug', type=str, default=DATA_AUG, help='Data augmentation methods')

    flags = parser.parse_args()
    flags.input_size = (flags.input_size, flags.input_size)
    return flags


def main(flags):
    # set gpu
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = flags.GPU
    # environment settings
    np.random.seed(flags.random_seed)
    tf.set_random_seed(flags.random_seed)

    # data prepare step
    Data = rsrClassData(flags.rsr_data_dir)
    (collect_files_train, meta_train) = Data.getCollectionByName(flags.train_data_dir)
    pe_train = patch_extractor.PatchExtractorInria(flags.rsr_data_dir,
                                                   collect_files_train, patch_size=flags.input_size,
                                                   tile_dim=meta_train['dim_image'][:2],
                                                   appendix=flags.train_patch_appendix,
                                                   overlap=184)
    train_data_dir = pe_train.extract(flags.patch_dir, pad=184)
    (collect_files_valid, meta_valid) = Data.getCollectionByName(flags.valid_data_dir)
    pe_valid = patch_extractor.PatchExtractorInria(flags.rsr_data_dir,
                                                   collect_files_valid, patch_size=flags.input_size,
                                                   tile_dim=meta_valid['dim_image'][:2],
                                                   appendix=flags.valid_patch_appendix,
                                                   overlap=184)
    valid_data_dir = pe_valid.extract(flags.patch_dir, pad=184)

    # image reader
    coord = tf.train.Coordinator()

    # load reader
    with tf.name_scope('image_loader'):
        reader_train = image_reader.ImageLabelReader(train_data_dir, flags.input_size, coord,
                                                     city_list=flags.city_name, tile_list=flags.train_tile_names,
                                                     data_aug=flags.data_aug)
        reader_valid = image_reader.ImageLabelReader(valid_data_dir, flags.input_size, coord,
                                                     city_list=flags.city_name, tile_list=flags.valid_tile_names,
                                                     data_aug=flags.data_aug)
        X_batch_op, y_batch_op = reader_train.dequeue(flags.batch_size)
        X_batch_op_valid, y_batch_op_valid = reader_valid.dequeue(flags.batch_size)
    reader_train_op = [X_batch_op, y_batch_op]
    reader_valid_op = [X_batch_op_valid, y_batch_op_valid]

    # define place holder
    X = tf.placeholder(tf.float32, shape=[None, flags.input_size[0], flags.input_size[1], 3], name='X')
    y = tf.placeholder(tf.int32, shape=[None, flags.input_size[0], flags.input_size[1], 1], name='y')
    mode = tf.placeholder(tf.bool, name='mode')

    # initialize model
    flags.model_name = '{}_EP-{}_DS-{}_LR-{}'.format(flags.model_name, flags.epochs, flags.decay_step, flags.learning_rate)
    model = unet.UnetModel_Height_Appendix({'X':X, 'Y':y}, trainable=mode, model_name=flags.model_name, input_size=flags.input_size)
    model.create_graph('X', flags.num_classes, start_filter_num=40)
    model.make_loss('Y')
    model.make_learning_rate(flags.learning_rate,
                             tf.cast(flags.n_train/flags.batch_size * flags.decay_step, tf.int32), flags.decay_rate)
    model.make_update_ops('X', 'Y')
    model.make_optimizer(model.learning_rate)
    # set ckdir
    model.make_ckdir(flags.ckdir)
    # make summary
    model.make_summary()
    # set up graph and initialize
    config = tf.ConfigProto()

    # run training
    start_time = time.time()
    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)

        if os.path.exists(model.ckdir) and tf.train.get_checkpoint_state(model.ckdir):
            latest_check_point = tf.train.latest_checkpoint(model.ckdir)
            saver.restore(sess, latest_check_point)

        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        try:
            train_summary_writer = tf.summary.FileWriter(model.ckdir, sess.graph)

            model.train('X', 'Y', flags.epochs, flags.n_train, flags.batch_size, sess, train_summary_writer,
                        train_reader=reader_train_op, valid_reader=reader_valid_op, image_summary=utils.image_summary)
        finally:
            coord.request_stop()
            coord.join(threads)
            saver.save(sess, '{}/model.ckpt'.format(model.ckdir), global_step=model.global_step)

    duration = time.time() - start_time
    print('duration {:.2f} hours'.format(duration/60/60))

if __name__ == '__main__':
    flags = read_flag()
    main(flags)
