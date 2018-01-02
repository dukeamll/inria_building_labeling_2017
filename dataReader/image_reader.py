import os
import re
import scipy.misc
import numpy as np
import tensorflow as tf
from dataReader import patch_extractor


def block_flipping(block):
    return tf.image.random_flip_left_right(tf.image.random_flip_up_down(block))


def block_rotating(block):
    random_times = tf.to_int32(tf.random_uniform([1], minval=0, maxval=4))[0]
    return tf.image.rot90(block, random_times)


def image_flipping(img, label):
    """
    randomly flips images left-right and up-down
    :param img:
    :param label:
    :return:flipped images
    """
    label = tf.cast(label, dtype=tf.float32)
    temp = tf.concat([img, label], axis=2)
    temp_flipped = block_flipping(temp)
    img = tf.slice(temp_flipped, [0, 0, 0], [-1, -1, 3])
    label = tf.slice(temp_flipped, [0, 0, 3], [-1, -1, 1])
    return img, label


def image_rotating(img, label):
    """
    randomly rotate images by 0/90/180/270 degrees
    :param img:
    :param label:
    :return:rotated images
    """
    temp = tf.concat([img, label], axis=2)
    temp_rotated = block_rotating(temp)
    img = tf.slice(temp_rotated, [0, 0, 0], [-1, -1, 3])
    label = tf.slice(temp_rotated, [0, 0, 3], [-1, -1, 1])
    return img, label


def read_images_labels_from_disk(input_queue, input_size, data_aug='', image_mean=0):
    image_contents = tf.read_file(input_queue[0])
    label_contents = tf.read_file(input_queue[1])

    image = tf.image.decode_jpeg(image_contents, channels=3)
    image = tf.cast(image, dtype=tf.float32)
    image -= image_mean
    # adhoc decoding for labels
    label = tf.image.decode_png(label_contents, channels=1)/255

    if 'flip' in data_aug:
        image, label = image_flipping(image, label)
    if 'rotate' in data_aug:
        image, label = image_rotating(image, label)

    image.set_shape((input_size[0], input_size[1], 3))
    label.set_shape((input_size[0], input_size[1], 1))

    return image, label


def image_label_iterator(image_dir, batch_size, tile_dim, patch_size, overlap=0, padding=0, image_mean=0):
    # this is a iterator for test
    block = scipy.misc.imread(image_dir)
    if padding > 0:
        block = patch_extractor.pad_block(block, padding)
        tile_dim = (tile_dim[0]+padding*2, tile_dim[1]+padding*2)
    cnt = 0
    image_batch = np.zeros((batch_size, patch_size[0], patch_size[1], 3))
    for patch in patch_extractor.patchify(block, tile_dim, patch_size, overlap=overlap):
        cnt += 1
        image_batch[cnt-1, :, :, :] = patch
        if cnt == batch_size:
            cnt = 0
            yield image_batch - image_mean
    # yield the last chunck
    if cnt > 0:
        yield image_batch[:cnt, :, :, :] - image_mean


def read_batch_from_list(file_list, batch_idx):
    block = []
    for idx in batch_idx:
        if file_list[idx][-3:] != 'npy':
            img = scipy.misc.imread(file_list[idx])
        else:
            img = np.load(file_list[idx])
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        block.append(img)
    return np.stack(block, axis=0)


class ImageLabelReader(object):
    def __init__(self, data_dir, input_size, coord, city_list, tile_list,
                 data_list='data_list.txt', random=True, ds_name='inria',
                 data_aug='', image_mean=0):
        self.original_dir = ''
        self.data_dir = data_dir
        self.data_list = data_list
        self.ds_name = ds_name
        self.data_aug = data_aug
        self.image_mean = image_mean
        self.input_size = input_size
        self.coord = coord
        self.city_list = city_list
        self.tile_list = tile_list
        self.image_list, self.label_list = self.read_image_label_list()
        self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
        self.labels = tf.convert_to_tensor(self.label_list, dtype=tf.string)
        self.queue = tf.train.slice_input_producer([self.images, self.labels], shuffle=random)
        self.image, self.label = read_images_labels_from_disk(self.queue, self.input_size, self.data_aug, self.image_mean)

    def read_image_label_list(self):
        with open(os.path.join(self.data_dir, self.data_list), 'r') as file:
            files = file.readlines()
        image_list = []
        label_list = []
        for file in files:
            file_tuple = file.strip('\n').split(' ')
            if self.ds_name == 'inria':
                city_name = re.findall('^[a-z\-]*', file_tuple[0])[0]
                tile_id = re.findall('[0-9]+(?=_img)', file_tuple[0])[0]
            else:
                city_name = file_tuple[0][:3]
                tile_id = file_tuple[0][3:6].lstrip('0')
            if city_name in self.city_list and tile_id in self.tile_list:
                image_list.append(os.path.join(self.data_dir, file_tuple[0]))
                label_list.append(os.path.join(self.data_dir, file_tuple[1]))
        if len(image_list) == 0:
            raise ValueError
        return image_list, label_list

    def set_original_image_label_dir(self, origin_dir):
        self.original_dir = origin_dir

    def dequeue(self, num_elements):
        image_batch, label_batch = tf.train.batch([self.image, self.label], num_elements)
        return image_batch, label_batch
