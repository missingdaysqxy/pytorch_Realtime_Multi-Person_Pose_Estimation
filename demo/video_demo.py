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

VIDEO_EXT = ['.mp4', '.avi', '.mpg', '.mpeg', '.mov']

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


def organize_video_io_paths(input_data, output_dir, output_format):
    io_paths = {"input": [], "output": []}
    if not os.path.exists(input_data):
        raise FileNotFoundError("File not exist in {}".format(input_data))
    if os.path.isdir(input_data):
        for root, dirs, files in os.walk(input_data):
            rel_path = os.path.relpath(root, input_data)
            for file in files:
                name, ext = os.path.splitext(file)
                if ext.lower() in VIDEO_EXT:
                    input_path = os.path.join(root, file)
                    output_path = os.path.join(output_dir, rel_path, name + output_format)
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    io_paths["input"].append(input_path)
                    io_paths["output"].append(output_path)
                else:
                    warn("Unsupported format: %s" % file)
    else:
        name, ext = os.path.splitext(input_data)
        assert ext.lower() in VIDEO_EXT, "Unsupported format: %s" % input_data
        output_path = os.path.join(output_dir, os.path.basename(name) + output_format)
        io_paths["input"].append(input_data)
        io_paths["output"].append(output_path)
    return io_paths


def main(args):
    input_data = args.video
    weight_file = args.weight
    frame_rate_ratio = args.frame_ratio
    process_speed = args.process_speed
    resize_fac = args.resize_factor
    output_dir = 'outputs/video'
    output_format = '.mp4'

    print('start processing...')

    # Video input & output

    io_paths = organize_video_io_paths(input_data, output_dir, output_format)

    # load model
    print('[*]Loading model...')
    model = get_model('vgg19')
    model.load_state_dict(torch.load(weight_file))
    model = torch.nn.DataParallel(model).cuda()
    model.float()
    model.eval()

    # Video reader
    for input_path, output_path in zip(io_paths["input"], io_paths["output"]):
        print('[*]Process video {} into {}'.format(input_path, output_path))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cam = cv2.VideoCapture(input_path)
        input_fps = cam.get(cv2.CAP_PROP_FPS)

        # ret_val, input_image = cam.read()
        # if not ret_val:
        #     continue
        video_length = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))

        ending_frame = args.out_length
        if ending_frame is None:
            ending_frame = video_length

        # Video writer
        output_fps = input_fps / frame_rate_ratio
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height = int(resize_fac * input_data.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(resize_fac * input_data.get(cv2.CAP_PROP_FRAME_WIDTH))
        out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
        i = 0  # default is 0
        t0 = time.time()
        while (cam.isOpened()) and i < ending_frame:
            ret_val, input_image = cam.read()
            if not ret_val:
                break
            if i % frame_rate_ratio == 0:
                t1 = time.time()
                # generate image with body parts
                resized_image = cv2.resize(input_image, (0, 0), fx=1 * resize_fac, fy=1 * resize_fac,
                                           interpolation=cv2.INTER_CUBIC)
                to_plot, canvas, joint_list, person_to_joint_assoc = process(model, resized_image, process_speed)
                if args.verb:
                    cv2.imshow('preview', to_plot)
                    cv2.waitKey(1)
                t2 = time.time()
                processBar(i, ending_frame,
                           '{}/{}, process time:{:.3f}, total time:{:.3f}'.format(i, ending_frame, (t2 - t1),
                                                                                  (t2 - t0)), length=20)
                out.write(canvas)
            i += 1
        out.release()
        cv2.destroyAllWindows()
        processBar(ending_frame, ending_frame,
                   '{}/{}, total time:{:.3f}'.format(i, ending_frame, (time.time() - t0)),
                   length=20)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video', type=str, help='input video file name')
    parser.add_argument('-w', '--weight', type=str, default='./network/weight/pose_model.pth',
                        help='path to the weights file')
    parser.add_argument('-f', '--resize_factor', type=float, default=1, help='minification factor')
    parser.add_argument('-r', '--frame_ratio', type=int, default=1, help='analyze every [n] frames')
    parser.add_argument('-s', '--process_speed', type=int, default=2,
                        help='Int 1 (fastest, lowest quality) to 4 (slowest, highest quality)')
    parser.add_argument('-l', '--out_length', type=int, default=None, help='Last video frame to analyze')
    parser.add_argument('-v', '--verb', action='store_true', help='show video canvas')
    args = parser.parse_args()
    main(args)
