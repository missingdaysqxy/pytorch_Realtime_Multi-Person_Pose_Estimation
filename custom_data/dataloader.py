# coding=utf-8
import os
import cv2
import math
import numpy as np

import torch
from pycocotools.coco import COCO
from scipy.spatial.distance import cdist

from custom_data.heatmap import putGaussianMaps
from custom_data.paf import putVecMaps
from custom_data.preprocessing import (inception_preprocess,
                                       rtpose_preprocess,
                                       ssd_preprocess, vgg_preprocess)
from torch.utils.data import DataLoader, Dataset



class DidiDataset(Dataset):
    idx_in_coco_str = ['left_eye', 'right_eye', 'nose', 'neck', 'left_chest', 'right_chest',
                       'left_shoulder', 'left_upperarm', 'left_elbow', 'left_forearm', 'left_wrist', 'left_hand',
                       'right_shoulder', 'right_upperarm', 'right_elbow', 'right_forearm', 'right_wrist', 'right_hand']

    num_joints = len(idx_in_coco_str)  # 18

    num_joints_and_bkg = num_joints + 1  # 19

    idx_in_coco = list(range(num_joints))  # [0:17]

    joint_pairs = [[3, 2], [2, 0], [2, 1], [3, 4], [3, 5],
                   [3, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11],
                   [3, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17]]

    num_connections = len(joint_pairs)  # 17

    def __init__(self, img_dir, anno_path, target_size=(368,368),stride=8):
        self.coco_anno = COCO(anno_path)
        self.img_dir = img_dir
        self.ids = list(self.coco_anno.imgs.keys())

        for i, idx in enumerate(self.ids):
            img_meta = self.coco_anno.imgs[idx]

            # load annotations

            id = img_meta['id']
            img_file = img_meta['file_name']
            h, w = img_meta['height'], img_meta['width']
            img_path = os.path.join(self.img_dir, img_file)
            ann_ids = self.coco_anno.getAnnIds(imgIds=id)
            anns = self.coco_anno.loadAnns(ann_ids)

            total_keypoints = sum([ann.get('num_keypoints', 0) for ann in anns])
            if total_keypoints == 0:
                continue

            persons = []
            prev_center = []
            masks = []
            keypoints = []

            # sort from the biggest person to the smallest one
            persons_ids = np.argsort([-a['area'] for a in anns], kind='mergesort')

            for id in list(persons_ids):
                person_meta = anns[id]

                if person_meta["iscrowd"]:
                    masks.append(self.coco_anno.annToRLE(person_meta))
                    continue

                # skip this person if parts number is too low or if segmentation area is too small
                if person_meta["num_keypoints"] < 5 or person_meta["area"] < 32 * 32:
                    masks.append(self.coco_anno.annToRLE(person_meta))
                    continue

                # skip this person if the distance to existing person is too small
                person_center = [person_meta["bbox"][0] + person_meta["bbox"][2] / 2,
                                 person_meta["bbox"][1] + person_meta["bbox"][3] / 2]
                too_close = False
                for pc in prev_center:
                    a = np.expand_dims(pc[:2], axis=0)
                    b = np.expand_dims(person_center, axis=0)
                    dist = cdist(a, b)[0]
                    if dist < pc[2] * 0.3:
                        too_close = True
                        break
                if too_close:
                    # add mask of this person. we don't want to show the network unlabeled people
                    masks.append(self.coco_anno.annToRLE(person_meta))
                    continue

                keypoints.append(person_meta["keypoints"])
                pers = PersonMeta(
                    img_path=img_path,
                    height=h,
                    width=w,
                    center=np.expand_dims(person_center, axis=0),
                    bbox=person_meta["bbox"],
                    area=person_meta["area"],
                    scale=person_meta["bbox"][3] / target_size[0],
                    num_keypoints=person_meta["num_keypoints"])
                persons.append(pers)
                prev_center.append(np.append(person_center, max(person_meta["bbox"][2],
                                                                person_meta["bbox"][3])))

            if len(persons) > 0:
                main_person = persons[0]
                main_person.masks_segments = masks
                main_person.all_joints = DidiDataset.from_coco_keypoints(keypoints, w, h)
                self.all_meta.append(main_person)

            if i % 1000 == 0:
                print("Loading image annot {}/{}".format(i, len(ids)))


    def get_ground_truth(self):
        # create heatmap
        heatmap = DidiDataset.create_heatmap()
        paf=DidiDataset.create_paf()


    @staticmethod
    def from_coco_keypoints(all_keypoints, w, h):
        """
        Creates list of joints based on the list of coco keypoints vectors.
        :param all_keypoints: list of coco keypoints vector [[x1,y1,v1,x2,y2,v2,....], []]
        :param w: image width
        :param h: image height
        :return: list of joints [[(x1,y1), (x1,y1), ...], [], []]
        """
        all_joints = []
        for keypoints in all_keypoints:
            kp = np.array(keypoints)
            xs = kp[0::3]
            ys = kp[1::3]
            vs = kp[2::3]

            # filter and loads keypoints to the list
            keypoints_list = []
            for idx, (x, y, v) in enumerate(zip(xs, ys, vs)):
                # only visible and occluded keypoints are used
                if v >= 1 and x >= 0 and y >= 0 and x < w and y < h:
                    keypoints_list.append((x, y))
                else:
                    keypoints_list.append(None)

            # build the list of joints. It contains the same coordinates
            # of body parts like in the orginal coco keypoints plus
            # additional body parts interpolated from coco
            # keypoints (ex. a neck)
            joints = []
            for part_idx in range(len(DidiDataset.idx_in_coco)):
                coco_kp_idx = DidiDataset.idx_in_coco[part_idx]
                if callable(coco_kp_idx):
                    p = coco_kp_idx(keypoints_list)
                else:
                    p = keypoints_list[coco_kp_idx]
                joints.append(p)
            all_joints.append(joints)
        return all_joints

    @staticmethod
    def create_heatmap(num_maps, height, width, all_joints, sigma, stride):
        def _put_heatmap_on_plane(heatmap, plane_idx, joint, sigma, height, width, stride):
            start = stride / 2.0 - 0.5
            center_x, center_y = joint
            for g_y in range(height):
                for g_x in range(width):
                    x = start + g_x * stride
                    y = start + g_y * stride
                    d2 = (x - center_x) * (x - center_x) + (y - center_y) * (y - center_y)
                    exponent = d2 / 2.0 / sigma / sigma
                    if exponent > 4.6052:
                        continue
                    heatmap[g_y, g_x, plane_idx] += math.exp(-exponent)
                    if heatmap[g_y, g_x, plane_idx] > 1.0:
                        heatmap[g_y, g_x, plane_idx] = 1.0
        heatmap = np.zeros((height, width, num_maps), dtype=np.float64)
        for joints in all_joints:
            for plane_idx, joint in enumerate(joints):
                if joint:
                    _put_heatmap_on_plane(heatmap, plane_idx, joint, sigma, height, width, stride)
        # background
        heatmap[:, :, -1] = np.clip(1.0 - np.amax(heatmap, axis=2), 0.0, 1.0)
        return heatmap

    @staticmethod
    def create_paf(num_maps, height, width, all_joints, threshold, stride):
        def _put_paf_on_plane(vectormap, countmap, plane_idx, x1, y1, x2, y2, threshold, height, width):
            min_x = max(0, int(round(min(x1, x2) - threshold)))
            max_x = min(width, int(round(max(x1, x2) + threshold)))
            min_y = max(0, int(round(min(y1, y2) - threshold)))
            max_y = min(height, int(round(max(y1, y2) + threshold)))

            vec_x = x2 - x1
            vec_y = y2 - y1
            norm = math.sqrt(vec_x ** 2 + vec_y ** 2)
            if norm < 1e-8:
                return

            vec_x /= norm
            vec_y /= norm

            for y in range(min_y, max_y):
                for x in range(min_x, max_x):
                    bec_x = x - x1
                    bec_y = y - y1
                    dist = abs(bec_x * vec_y - bec_y * vec_x)
                    if dist > threshold:
                        continue
                    cnt = countmap[y][x][plane_idx]
                    if cnt == 0:
                        vectormap[y][x][plane_idx * 2 + 0] = vec_x
                        vectormap[y][x][plane_idx * 2 + 1] = vec_y
                    else:
                        vectormap[y][x][plane_idx * 2 + 0] = (vectormap[y][x][plane_idx * 2 + 0] * cnt + vec_x) / (cnt + 1)
                        vectormap[y][x][plane_idx * 2 + 1] = (vectormap[y][x][plane_idx * 2 + 1] * cnt + vec_y) / (cnt + 1)
                    countmap[y][x][plane_idx] += 1

        paf = np.zeros((height, width, num_maps * 2), dtype=np.float64)
        countmap = np.zeros((height, width, num_maps), dtype=np.uint8)
        for joints in all_joints:
            for plane_idx, (j_idx1, j_idx2) in enumerate(DidiDataset.joint_pairs):
                center_from = joints[j_idx1]
                center_to = joints[j_idx2]
                # skip if no valid pair of keypoints
                if center_from is None or center_to is None:
                    continue
                x1, y1 = (center_from[0] / stride, center_from[1] / stride)
                x2, y2 = (center_to[0] / stride, center_to[1] / stride)
                _put_paf_on_plane(paf, countmap, plane_idx, x1, y1, x2, y2,
                                  threshold, height, width)
        return paf


class PersonMeta(object):
    """
    PersonMeta representing a single data point for training.
    """
    __slots__ = (
        'img_path',
        'height',
        'width',
        'center',
        'bbox',
        'area',
        'num_keypoints',
        'masks_segments',
        'scale',
        'all_joints',
        'img',
        'mask',
        'aug_center',
        'aug_joints')

    def __init__(self, img_path, height, width, center, bbox,
                 area, scale, num_keypoints):
        self.img_path = img_path
        self.height = height
        self.width = width
        self.center = center
        self.bbox = bbox
        self.area = area
        self.scale = scale
        self.num_keypoints = num_keypoints

        # updated after iterating over all persons
        self.masks_segments = None
        self.all_joints = None

        # updated during augmentation
        self.img = None
        self.mask = None
        self.aug_center = None
        self.aug_joints = None
