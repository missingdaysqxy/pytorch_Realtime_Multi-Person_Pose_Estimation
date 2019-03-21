# import ptvsd
# print("wait for vscode attachment...")
# ptvsd.enable_attach(address=("0.0.0.0", 8078))
# ptvsd.wait_for_attach()

import os
import sys
import cv2
import h5py
import time
import json
import argparse
import torch
import numpy as np
from torchnet import meter
from pycocotools.coco import COCO
from warnings import warn

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append(os.path.abspath(os.path.curdir))
from network.rtpose_vgg import get_model
from network.post import decode_pose
from evaluate.coco_eval import get_multiplier, get_outputs, handle_paf_and_heat

IMAGE_EXT = ['.jpg', '.png', '.bmp', '.jpeg', '.jpe', '.tif', '.tiff']


def processBar(num, total, msg='', length=50, end=None):
    rate = num / total
    rate_num = int(rate * 100)
    clth = int(rate * length)
    if len(msg) > 0:
        msg += ':'
    if rate_num == 100:
        r = '\r%s[%s%d%%]%s\n' % (msg, '*' * length, rate_num, end)
    else:
        r = '\r%s[%s%s%d%%]%s' % (msg, '*' * clth, '-' * (length - clth), rate_num, end)
    sys.stdout.write(r)
    sys.stdout.flush
    return r.replace('\r', ':')


def process(model, oriImg, process_speed):
    # Get results of original image
    multiplier = get_multiplier(oriImg, process_speed)
    with torch.no_grad():
        orig_paf, orig_heat = get_outputs(
            multiplier, oriImg, model, 'rtpose')

        # Get results of flipped image
        swapped_img = oriImg[:, ::-1, :]
        flipped_paf, flipped_heat = get_outputs(multiplier, swapped_img,
                                                model, 'rtpose')

        # compute averaged heatmap and paf
        paf, heatmap = handle_paf_and_heat(
            orig_heat, flipped_heat, orig_paf, flipped_paf)
    param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
    to_plot, canvas, joint_list, person_to_joint_assoc = decode_pose(
        oriImg, param, heatmap, paf)
    return to_plot, canvas, joint_list, person_to_joint_assoc


def get_data(data_dir, json_path):
    if not os.path.exists(data_dir):
        raise FileNotFoundError("File not exist in {}".format(data_dir))
    if not os.path.isfile(json_path):
        raise FileNotFoundError("File not exist in {}".format(json_path))
    kp_didi2this = [15, 14, 0, 1, -1, -1, 5, -1, 6, -1, 7, -1, 2, -1, 3, -1, 4, -1]
    coco = COCO(json_path)
    rets=[]
    for i, (img_id, img_info) in enumerate(coco.imgs.items()):
        image_path = os.path.join(data_dir, img_info["file_name"])
        if not os.path.isfile(image_path):
            warn("Image not exist in {}".format(image_path))
            continue
        anns = coco.loadAnns(coco.getAnnIds(img_id))
        keypoints = []
        for ann in anns:
            ann_kp = np.array(ann['keypoints']).reshape((-1, 3))
            new_kp = np.zeros_like(ann_kp)
            valid_kp_count = 0
            for i, idx in enumerate(kp_didi2this):
                if kp_didi2this[idx] >= 0:
                    new_kp[kp_didi2this[idx]] = ann_kp[i]
                    if ann_kp[i, 2] != 0:
                        valid_kp_count += 1
            keypoints.append([valid_kp_count, new_kp])
        rets.append(image_path, keypoints)
    return rets

def main(args):
    weight_file = args.weight
    process_speed = args.process_speed
    resize_fac = args.resize_factor

    print('start processing...')

    # Video input & output
    data_dir = args.dir
    json_path = args.coco

    # load model
    print('[*]Loading model...')
    model = get_model('vgg19')
    model.load_state_dict(torch.load(weight_file))
    model = torch.nn.DataParallel(model).cuda()
    model.float()
    model.eval()
    print('Model Ready!')

    # Video reader
    t0 = time.time()
    acc_count = 0
    data=get_data(data_dir, json_path)
    data_count = len(data)
    for i, (input_path, keypoints) in enumerate(data):
        input_image = cv2.imread(input_path)
        t1 = time.time()
        # generate image with body parts
        resized_image = cv2.resize(input_image, (0, 0), fx=1 * resize_fac, fy=1 * resize_fac,
                                   interpolation=cv2.INTER_CUBIC)
        to_plot, canvas, joint_list, person_to_joint_assoc = process(model, resized_image, process_speed)
        kp_count = 0
        for c, kps in keypoints:
            kp_count += c
        if len(person_to_joint_assoc) == len(keypoints):  # human count equal
            acc_count += 1
        if args.verb:
            cv2.imshow('preview', to_plot)
            cv2.waitKey(1)
        t2 = time.time()
        processBar(i, data_count, '[{}/{}]find {} keypoints in {} humans, groundtruth is {} kps in {} humans. acc:{} process time:{:.3f}, total time:{:.3f}'.format(
            i, data_count,len(joint_list), len(person_to_joint_assoc), kp_count, len(keypoints),
             acc_count / (i + 1), (t2 - t1), (t2 - t0)), length=20, end="\n")
    cv2.destroyAllWindows()
    processBar(data_count, data_count, '{}/{}, acc:{} total time:{:.3f}'.format(
        data_count, data_count, acc_count / data_count, (time.time() - t0)), length=20)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str, help='data dir path')
    parser.add_argument('coco', type=str, help="coco json file")
    parser.add_argument('-w', '--weight', type=str, default='./network/weight/pose_model.pth',
                        help='path to the weights file')
    parser.add_argument('-f', '--resize_factor', type=float, default=1, help='minification factor')
    parser.add_argument('-s', '--process_speed', type=int, default=4,
                        help='Int 1 (fastest, lowest quality) to 4 (slowest, highest quality)')
    parser.add_argument('-v', '--verb', action='store_true', help='show video canvas')
    args = parser.parse_args()
    main(args)
