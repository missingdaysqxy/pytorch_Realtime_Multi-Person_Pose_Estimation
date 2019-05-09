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
import h5py

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.append(os.path.abspath(os.path.curdir))
from network.rtpose_vgg import get_model
from network.post import decode_pose
from evaluate.coco_eval import get_multiplier, get_outputs, handle_paf_and_heat
from utils import organize_1to1_io_paths, organize_Nto1_io_paths, processBar

IMAGE_EXT = ['.jpg', '.png', '.bmp', '.jpeg', '.jpe', '.tif', '.tiff']


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
    process_speed = args.process_speed
    resize_fac = args.resize_factor
    output_dir = args.output
    output_format = '.h5'
    save_demo = args.verb

    print('start processing...')

    # Video input & output
    if args.input_type == 'serial':
        io_paths = organize_Nto1_io_paths(input_data, IMAGE_EXT, output_dir, output_format)
    else:
        io_paths = organize_1to1_io_paths(input_data, IMAGE_EXT, output_dir, output_format)
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
        if io_paths["type"] == "1to1":
            print('[*]Process {} into {}'.format(input_path, output_path))
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            input_image = cv2.imread(input_path)
            out_h5 = h5py.File(output_path, mode="w")
            out_h5["height"] = input_image.shape[0]
            out_h5["width"] = input_image.shape[1]
            t1 = time.time()
            # generate image with body parts
            resized_image = cv2.resize(input_image, (0, 0), fx=1 * resize_fac, fy=1 * resize_fac,
                                       interpolation=cv2.INTER_CUBIC)
            to_plot, canvas, joint_list, person_to_joint_assoc = process(model, resized_image, process_speed)
            frame_h5 = out_h5.create_group("frame0")
            frame_h5.create_dataset("joint_list", data=joint_list)
            frame_h5.create_dataset("person_to_joint_assoc", data=person_to_joint_assoc)
            if save_demo:
                demo_path = os.path.splitext(output_path)[0] + ".jpg"
                cv2.imwrite(demo_path, canvas)
            t2 = time.time()
            processBar(i, data_count,
                       '{}/{}, process time:{:.3f}, total time:{:.3f}'.format(i, data_count, (t2 - t1),
                                                                              (t2 - t0)), length=20)
        elif len(input_path[0]) > 0:
            print('[*]Process {} into {}'.format(os.path.dirname(input_path[0]), output_path))
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            input_image = cv2.imread(input_path[0])
            height = input_image.shape[0]
            width = input_image.shape[1]
            out_h5 = h5py.File(output_path, mode="w")
            out_h5["height"] = height
            out_h5["width"] = width
            if save_demo:  # Video writer
                demo_path = os.path.splitext(output_path)[0] + ".mp4"
                output_fps = 15
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_demo = cv2.VideoWriter(demo_path, fourcc, output_fps, (width, height))
            i = 0  # default is 0
            t0 = time.time()
            for j, path in enumerate(input_path):
                input_image = cv2.imread(path)
                t1 = time.time()
                # generate image with body parts
                resized_image = cv2.resize(input_image, (0, 0), fx=1 * resize_fac, fy=1 * resize_fac,
                                           interpolation=cv2.INTER_CUBIC)
                to_plot, canvas, joint_list, person_to_joint_assoc = process(model, resized_image, process_speed)
                frame_h5 = out_h5.create_group("frame%d" % j)
                frame_h5.create_dataset("joint_list", data=joint_list)
                frame_h5.create_dataset("person_to_joint_assoc", data=person_to_joint_assoc)
                if save_demo:
                    out_demo.write(canvas)
                t2 = time.time()
                processBar(j, len(input_path),
                           '{}/{}, process time:{:.3f}, total time:{:.3f}'.format(j, len(input_path), (t2 - t1),
                                                                                  (t2 - t0)), length=20)
            if save_demo:
                out_demo.release()
            out_h5.close()
            processBar(len(input_path), len(input_path),
                       '{}/{}, total time:{:.3f}'.format(j, len(input_path), (time.time() - t0)),
                       length=45)

        cv2.destroyAllWindows()
        processBar(data_count, data_count,
                   '{}/{}, total time:{:.3f}'.format(i, data_count, (time.time() - t0)), length=45)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='input image file or directory')
    parser.add_argument('-o', '--output', type=str, default='outputs',
                        help='output directory to save h5py file(s)')
    parser.add_argument('-t', '--input_type', choices=['serial', 'lone'], default='serial',
                        help='serial: Treat image sequence as frames of a video;\n'
                             'lone: Treat all images independently. Default: serial')
    parser.add_argument('-w', '--weight', type=str, default='./network/weight/pose_model.pth',
                        help='path to the weights file')
    parser.add_argument('-f', '--resize_factor', type=float, default=1, help='minification factor')
    parser.add_argument('-s', '--process_speed', type=int, default=2,
                        help='Int 1 (fastest, lowest quality) to 4 (slowest, highest quality)')
    parser.add_argument('-v', '--verb', action='store_true',
                        help='FLAG: save demo into output directory or not,'
                             ' the demo type is mp4 video if serial or jpg image if lone.')
    args = parser.parse_args()
    main(args)
