import os
import pickle
from glob import glob


class rsrClassData(object):
    def __init__(self, ds_dir, preproc=None, patch_extractor=None):
        self.ds_dir = ds_dir
        self.collection_list = self.getCollectionList()
        self.preproc = preproc
        self.patch_extractor = patch_extractor

    def getCollectionList(self):
        collections = glob(os.path.join(self.ds_dir, 'collections', '*'))
        return [collection.split('/')[-1] for collection in sorted(collections)]

    def getCollectionByName(self, name):
        collect_dir = os.path.join(self.ds_dir, 'collections')
        if not os.path.exists(os.path.join(collect_dir, name)):
            print('No such directory: {}'.format(os.path.join(collect_dir, name)))
            raise ValueError
        else:
            with open(os.path.join(collect_dir, name, '{}.collect'.format(name)), 'r') as collect_file:
                content = collect_file.readlines()
            collect_files = [c.strip('\n').split(' ') for c in content]
            with open(os.path.join(collect_dir, name, '{}_meta.pickle'.format(name)), 'rb') as collect_file:
                meta = pickle.load(collect_file)
            return collect_files, meta

    def getImage(self):
        # TODO get a tile
        image_list = []
        for c in self.getCollectionList():
            collect_files, _ = self.getCollectionByName(c)
            image_list.extend([collect_file[0] for collect_file in collect_files])
        return image_list

    def getImageTruth(self):
        # TODO get a tile
        truth_list = []
        for c in self.getCollectionList():
            collect_files, _ = self.getCollectionByName(c)
            if len(collect_files[0]) == 1:
                # no truth files available
                continue
            truth_list.extend([collect_file[1] for collect_file in collect_files])
        return truth_list
