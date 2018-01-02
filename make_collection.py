"""
This is an example of how to make collection
"""
import os
import re
import pickle
import scipy.misc
from glob import glob


def make_meta_inria(base_dir, image_files, truth_files=None, **kwargs):
    """
    This function can be anything as long as it returns a dictionary with
    its keys as attributes of metadata
    :return: dictionary of metadata
    """
    meta = dict()
    meta['tile number'] = len(image_files)
    if 'city' in kwargs:
        meta['city list'] = kwargs['city']
    else:
        meta['city list'] = 'austin,chicago,kitsap,tyrol-w,vienna'
    if 'tile' in kwargs:
        meta['tiles'] = kwargs['tile']
    else:
        meta['tiles'] = '1-36'
    colormap = {0:0, 1:255}
    meta['colormap'] = colormap
    meta['class num'] = 2
    meta['classes'] = 'background,building'
    meta['dim_image'] = scipy.misc.imread(os.path.join(base_dir, image_files[0])).shape
    if truth_files is not None:
        meta['dim_truth'] = scipy.misc.imread(os.path.join(base_dir, truth_files[0])).shape

    return meta


def make_collection_inria(name, data_dir, **kwargs):
    image_dir = os.path.join(data_dir, 'inria', 'image')
    truth_dir = os.path.join(data_dir, 'inria', 'truth')

    def get_files(dir, **kwargs):
        files = [file.replace(data_dir, '')[1:] for file in glob(os.path.join(dir, '*.tif'))]
        if 'city' in kwargs:
            files = [file for file in files if
                           re.findall('(?<=/)[a-z\-]*(?=[0-9]*.tif)', file)[0] in
                           kwargs['city'].split(',')]
        if 'tile' in kwargs:
            files = [file for file in files if
                           (int(kwargs['tile'].split('-')[0])
                            <= int(re.findall('[0-9]*(?=.tif)', file)[0])
                            <= int(kwargs['tile'].split('-')[1]))]
        return files

    # get images
    image_files = sorted(get_files(image_dir, **kwargs))
    print(image_files)
    # get truth
    truth_files = sorted(get_files(truth_dir, **kwargs))

    # make dir
    collect_dir = os.path.join(data_dir, 'collections', name)
    if not os.path.exists(collect_dir):
        os.makedirs(collect_dir)

    assert kwargs['mode'] == 'train' or kwargs['mode'] == 'test'
    if kwargs['mode'] == 'train':
        assert image_files[0].split('/')[-1] == truth_files[0].split('/')[-1]
        assert image_files[-1].split('/')[-1] == truth_files[-1].split('/')[-1]
        assert len(image_files) == len(truth_files)

        with open(os.path.join(collect_dir, '{}.collect'.format(name)), 'w') as collect_file:
            for i in range(len(image_files)):
                collect_file.write('{} {}\n'.format(image_files[i], truth_files[i]))
        meta = make_meta_inria(data_dir, image_files, truth_files, **kwargs)
    else:
        with open(os.path.join(collect_dir, '{}.collect'.format(name)), 'w') as collect_file:
            for i in range(len(image_files)):
                collect_file.write('{}\n'.format(image_files[i]))
        meta = make_meta_inria(data_dir, image_files, **kwargs)
    with open(os.path.join(collect_dir, '{}.meta'.format(name)), 'w') as collect_file:
        for key, val in meta.items():
            collect_file.write('{}:{}\n'.format(key, val))
    with open(os.path.join(collect_dir, '{}_meta.pickle'.format(name)), 'wb') as collect_file:
        pickle.dump(meta, collect_file)


def make_collection(name, data_dir, ds_name, **kwargs):
    if 'inria' in ds_name:
        make_collection_inria(name, data_dir, **kwargs)
