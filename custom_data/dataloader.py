"""MSCOCO Dataloader
   Thanks to @tensorboy @shuangliu
"""
try:
    import ujson as json
except ImportError:
    import json

from torchvision.transforms import ToTensor
from custom_data.data_pipeline import Cocokeypoints
from typing import Generator
from torch.utils.data.dataloader import DataLoader
import logging 

logger = logging.getLogger(__name__)



class sDataLoader(DataLoader):
    def get_stream(self):
        """
        Return a generate that can yield endless data.
        :Example:
        stream = get_stream()
        for i in range(100):
            batch = next(stream)

        :return: stream
        :rtype: Generator
        """
        while True:
            for data in iter(self):
                yield data

    @staticmethod
    def copy(loader):
        """
        Init a sDataloader from an existing Dataloader
        :param loader: an instance of Dataloader
        :type loader: DataLoader
        :return: a new instance of sDataloader
        :rtype: sDataLoader
        """
        if not isinstance(loader, DataLoader):
            logger.warning('loader should be an instance of Dataloader, but got {}'.format(type(loader)))
            return loader

        new_loader = sDataLoader(loader.dataset)
        for k, v in loader.__dict__.items():
            setattr(new_loader, k, v)
        return new_loader


def get_loader(json_path, data_dir, mask_dir, inp_size, feat_stride, preprocess,
               batch_size, params_transform, training=True, shuffle=True, num_workers=3):
    """ Build a COCO dataloader
    :param json_path: string, path to jso file
    :param datadir: string, path to coco data
    :returns : the data_loader
    """
    with open(json_path) as data_file:
        data_this = json.load(data_file)
        data = data_this['root']

    num_samples = len(data)
    train_indexes = []
    val_indexes = []
    for count in range(num_samples):
        if data[count]['isValidation'] != 0.:
            val_indexes.append(count)
        else:
            train_indexes.append(count)

    coco_data = Cocokeypoints(root=data_dir, mask_dir=mask_dir,
                              index_list=train_indexes if training else val_indexes,
                              data=data, inp_size=inp_size, feat_stride=feat_stride,
                              preprocess=preprocess, transform=ToTensor(), params_transform=params_transform)

    data_loader = sDataLoader(coco_data, batch_size=batch_size,
                              shuffle=shuffle, num_workers=num_workers)

    return data_loader
