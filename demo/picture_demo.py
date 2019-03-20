# import ptvsd
# print("wait for vscode attachment...")
# ptvsd.enable_attach(address=("0.0.0.0", 8078))
# ptvsd.wait_for_attach()

import os
import sys
import cv2
import time
import argparse
import torch
from warnings import warn

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.append(os.path.abspath(os.path.curdir))
from network.rtpose_vgg import get_model
from network.post import decode_pose
from evaluate.coco_eval import get_multiplier, get_outputs, handle_paf_and_heat

IMAGE_EXT = ['.jpg', '.png', '.bmp', '.jpeg', '.jpe', '.tif', '.tiff']


def processBar(num, total, msg='', length=50):
    rate = num / total
    rate_num = int(rate * 100)
    clth = int(rate * length)
    if len(msg) > 0:
        msg += ':'
    if rate_num == 100:
        r = '\r%s[%s%d%%]\n' % (msg, '*' * length, rate_num,)
    else:
        r = '\r%s[%s%s%d%%]' % (msg, '*' * clth, '-' * (length - clth), rate_num,)
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


def organize_image_io_paths(input_data, output_dir, output_ext=".jpg"):
    if not os.path.exists(input_data):
        raise FileNotFoundError("File not exist in {}".format(input_data))
    io_paths = {"input": [], "output": []}
    if os.path.isdir(input_data):
        for root, dirs, files in os.walk(input_data):
            rel_path = os.path.relpath(root, input_data)
            for file in files:
                name, ext = os.path.splitext(file)
                if ext.lower() in IMAGE_EXT:
                    input_path = os.path.join(root, file)
                    output_path = os.path.join(output_dir, rel_path, name + output_ext)
                    io_paths["input"].append(input_path)
                    io_paths["output"].append(output_path)
                else:
                    warn("Unsupported format: %s" % file)
    else:
        name, ext = os.path.splitext(input_data)
        assert ext.lower() in IMAGE_EXT, "Unsupported format: %s" % input_data
        output_path = os.path.join(output_dir, os.path.basename(name) + output_ext)
        io_paths["input"].append(input_data)
        io_paths["output"].append(output_path)
    return io_paths


def main(args):
    weight_file = args.weight
    process_speed = args.process_speed
    resize_fac = args.resize_factor

    print('start processing...')

    # Video input & output
    input_data = args.image
    output_dir = "outputs/images"
    io_paths = organize_image_io_paths(input_data, output_dir)
    data_count = len(io_paths["input"])

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
    for i, (input_path, output_path) in enumerate(zip(io_paths["input"], io_paths["output"])):
        print('[*]Process {} into {}'.format(input_path, output_path))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        input_image = cv2.imread(input_path)
        t1 = time.time()
        # generate image with body parts
        resized_image = cv2.resize(input_image, (0, 0), fx=1 * resize_fac, fy=1 * resize_fac,
                                   interpolation=cv2.INTER_CUBIC)
        to_plot, canvas, joint_list, person_to_joint_assoc = process(model, resized_image, process_speed)
        if args.verb:
            cv2.imshow('preview', to_plot)
            cv2.waitKey(1)
        t2 = time.time()
        processBar(i, data_count,
                   '{}/{}, process time:{:.3f}, total time:{:.3f}'.format(i, data_count, (t2 - t1),
                                                                          (t2 - t0)), length=20)
        cv2.imwrite(output_path, canvas)
    cv2.destroyAllWindows()
    processBar(data_count, data_count,
               '{}/{}, total time:{:.3f}'.format(i, data_count, (time.time() - t0)), length=20)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str, help='input image file name')
    parser.add_argument('-w', '--weight', type=str, default='./network/weight/pose_model.pth',
                        help='path to the weights file')
    parser.add_argument('-f', '--resize_factor', type=float, default=1, help='minification factor')
    parser.add_argument('-s', '--process_speed', type=int, default=2,
                        help='Int 1 (fastest, lowest quality) to 4 (slowest, highest quality)')
    parser.add_argument('-v', '--verb', action='store_true', help='show video canvas')
    args = parser.parse_args()
    main(args)
