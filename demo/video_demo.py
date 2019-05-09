# import ptvsd
# print("wait for vscode attachment...")
# ptvsd.enable_attach(address=("0.0.0.0", 8078))
# ptvsd.wait_for_attach()

import os
import sys
import cv2
import h5py
import time
import argparse
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.append(os.path.abspath(os.path.curdir))
from network.rtpose_vgg import get_model
from network.post import decode_pose
from evaluate.coco_eval import get_multiplier, get_outputs, handle_paf_and_heat
from utils import organize_1to1_io_paths, organize_Nto1_io_paths, processBar

VIDEO_EXT = ['.mp4', '.avi', '.mpg', '.mpeg', '.mov']
OUT_VDO_FMT = VIDEO_EXT[0]


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


def main(args):
    input_data = args.input
    weight_file = args.weight
    frame_rate_ratio = args.frame_ratio
    process_speed = args.process_speed
    resize_fac = args.resize_factor
    output_dir = args.output
    output_format = '.h5'
    save_demo = args.verb

    print('start processing...')

    # Video input & output

    io_paths = organize_1to1_io_paths(input_data, VIDEO_EXT, output_dir, output_format)

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

        # input video info
        cap = cv2.VideoCapture(input_path)
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        height = int(resize_fac * cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(resize_fac * cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        ending_frame = args.out_length
        if ending_frame is None:
            ending_frame = video_length

        out_h5 = h5py.File(output_path, mode="w")
        out_h5["height"] = height
        out_h5["width"] = width
        if save_demo:  # Video writer
            demo_path = os.path.splitext(output_path)[0] + ".mp4"
            output_fps = input_fps / frame_rate_ratio
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_demo = cv2.VideoWriter(demo_path, fourcc, output_fps, (width, height))
        i = 0  # default is 0
        t0 = time.time()
        while (cap.isOpened()) and i < ending_frame:
            ret_val, input_image = cap.read()
            if not ret_val:
                break
            if i % frame_rate_ratio == 0:
                t1 = time.time()
                # generate image with body parts
                resized_image = cv2.resize(input_image, (0, 0), fx=1 * resize_fac, fy=1 * resize_fac,
                                           interpolation=cv2.INTER_CUBIC)
                to_plot, canvas, joint_list, person_to_joint_assoc = process(model, resized_image, process_speed)
                frame_h5 = out_h5.create_group("frame%d" % i)
                frame_h5.create_dataset("joint_list", data=joint_list)
                frame_h5.create_dataset("person_to_joint_assoc", data=person_to_joint_assoc)
                if save_demo:
                    out_demo.write(canvas)
                t2 = time.time()
                processBar(i, ending_frame,
                           '{}/{}, process time:{:.3f}, total time:{:.3f}'.format(i, ending_frame, (t2 - t1),
                                                                                  (t2 - t0)), length=20)
            i += 1
        if save_demo:
            out_demo.release()
        out_h5.close()
        processBar(ending_frame, ending_frame,
                   '{}/{}, total time:{:.3f}'.format(i, ending_frame, (time.time() - t0)),
                   length=45)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='input video file or directory path')
    parser.add_argument('-o', '--output', type=str, default='outputs',
                        help='output directory to save h5py file(s)')
    parser.add_argument('-w', '--weight', type=str, default='./network/weight/pose_model.pth',
                        help='path to the weights file')
    parser.add_argument('-f', '--resize_factor', type=float, default=1, help='minification factor')
    parser.add_argument('-r', '--frame_ratio', type=int, default=1, help='analyze every [n] frames')
    parser.add_argument('-s', '--process_speed', type=int, default=2,
                        help='Int 1 (fastest, lowest quality) to 4 (slowest, highest quality)')
    parser.add_argument('-l', '--out_length', type=int, default=None, help='Last video frame to analyze')
    parser.add_argument('-v', '--verb', action='store_true',
                        help='FLAG: save mp4 video demo(s) into output directory or not')
    args = parser.parse_args()
    main(args)
