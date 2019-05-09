# coding: utf-8

# In[14]:


import os
import sys
import time
import argparse
import cv2
import h5py
import torch as t

sys.path.append(os.path.abspath(os.path.curdir))
from network.rtpose_vgg import get_model
from network.post import decode_pose
from evaluate.coco_eval import get_multiplier, get_outputs, handle_paf_and_heat
from utils import organize_1to1_io_paths,organize_Nto1_io_paths

## Paramenters & Constants
VIDEO_EXT = ['.mp4', '.avi', '.mpg', '.mpeg', '.mov']
IMAGE_EXT = ['.jpg', '.png', '.bmp', '.jpeg', '.jpe', '.tif', '.tiff']

## Select GPU Devices
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


## Main Process Function
def process(model, oriImg, process_speed):
    # Get results of original image
    multiplier = get_multiplier(oriImg, process_speed)
    with t.no_grad():
        orig_paf, orig_heat = get_outputs(multiplier, oriImg, model, 'rtpose')

        # Get results of flipped image
        swapped_img = oriImg[:, ::-1, :]
        flipped_paf, flipped_heat = get_outputs(multiplier, swapped_img, model, 'rtpose')

        # compute averaged heatmap and paf
        paf, heatmap = handle_paf_and_heat(orig_heat, flipped_heat, orig_paf, flipped_paf)
    param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
    to_plot, canvas, joint_list, person_to_joint_assoc = decode_pose(oriImg, param, heatmap, paf)
    return to_plot, canvas, joint_list, person_to_joint_assoc



## Data Loader
def load_video_frames(video_path, output_length=None, frame_rate_ratio=1):
    cam = cv2.VideoCapture(video_path)
    assert cam.isOpened(), "Open Video %s Failed!" % video_path
    video_length = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    if output_length is None:
        output_length = video_length
    i = 0  # default is 0
    while (cam.isOpened()) and i < output_length:
        ret_val, image = cam.read()
        if not ret_val:
            break
        if i % frame_rate_ratio == 0:
            yield image
        i += 1
    cam.release()


def get_video_size(video_path, output_length=None):
    cam = cv2.VideoCapture(video_path)
    assert cam.isOpened(), "Open Video %s Failed!" % video_path
    l = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    if output_length is not None:
        l = min(l, output_length)
    h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam.release()
    return l, h, w


def load_images_list(image_list, output_length=None, frame_rate_ratio=1):
    image_count = len(image_list)
    assert image_count > 0
    if output_length is None:
        output_length = image_count
    for i, path in enumerate(image_list):
        if i >= output_length:
            break
        if i % frame_rate_ratio == 0:
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            yield image


def get_images_size(image_list, output_length=None):
    image_count = len(image_list)
    assert image_count > 0
    image = cv2.imread(image_list[0])
    l = len(image_list)
    if output_length is not None:
        l = min(l, output_length)
    h, w = image.shape[:2]
    return l, h, w


## Calling Process
def main(args):
    input_data = args.input_dir
    input_type = args.input_type  # choose from ["image", "video"]
    output_dir = args.output_dir
    weight_file = args.weight
    input_ext = args.input_ext
    output_ext = args.out_ext
    frame_rate_ratio = args.frame_ratio  # analyze every [n] frames
    process_speed = args.process_speed  # int, 1 (fastest, lowest quality) to 4 (slowest, highest quality)
    resize_fac = args.resize_factor  # minification factor
    output_length = args.out_length  # int, frame count for output, None for input length
    show_visualize_process = args.verb  # show canvas through matplotlib
    rebuild_exist_file = args.rebuild

    ## Load Model
    model = get_model('vgg19')
    model.load_state_dict(t.load(weight_file))
    model = t.nn.DataParallel(model)
    model.cuda()
    model.float()
    model.eval()
    print("Model Ready!")

    ## Init I/O Paths
    _input_ext_ = IMAGE_EXT if input_ext == "image" \
        else VIDEO_EXT if input_ext == "video" \
        else input_ext if isinstance(input_ext, list) \
        else [input_ext]
    if input_type == "1to1":
        io_paths = organize_1to1_io_paths(input_data, _input_ext_, output_dir, output_ext)
    else:
        io_paths = organize_Nto1_io_paths(input_data, _input_ext_, output_dir, output_ext)
    total_item = len(io_paths["input"])
    print("Items count: ", total_item)

    ignore_item = 0
    for i, (input_dir, output_path) in enumerate(zip(io_paths["input"], io_paths["output"])):
        if os.path.isfile(output_path):
            if rebuild_exist_file:
                title = '[{}/{}]Rebuild {} from {}'
            else:
                print('[{}/{}]{} already exist, pass'.format(i, total_item, output_path))
                ignore_item += 1
                continue
        else:
            title = '[{}/{}]Build {} from {}'
        if isinstance(input_dir, str):  # process video
            source_position = input_dir
            loader = load_video_frames(input_dir, output_length, frame_rate_ratio)
            length, h, w = get_video_size(input_dir, output_length)
        elif isinstance(input_dir, list):  # process images
            source_position = os.path.dirname(input_dir[0])
            loader = load_images_list(input_dir, output_length, frame_rate_ratio)
            length, h, w = get_images_size(input_dir, output_length)
        else:
            raise TypeError("Expected string or list(string), but got %s" % type(input_dir))
        print(title.format(i, total_item, output_path, source_position))
        # Video writer
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            output_fps = 15
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            height = int(resize_fac * h)
            width = int(resize_fac * w)
            print("source:{}x{}  target:{}x{}".format(h, w, height, width))
            out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
            out_h5 = h5py.File(output_path + ".h5", mode="w")
            out_h5["height"] = height
            out_h5["width"] = width
            t0 = time.time()
            for i, frame in enumerate(loader):
                t1 = time.time()
                # generate image with body parts
                resized_image = cv2.resize(frame, (0, 0), fx=1 * resize_fac, fy=1 * resize_fac,
                                           interpolation=cv2.INTER_CUBIC)
                to_plot, canvas, joint_list, person_to_joint_assoc = process(model, resized_image, process_speed)
                # save outputs
                out.write(canvas)
                frame_h5 = out_h5.create_group("frame%d" % i)
                frame_h5.create_dataset("joint_list", data=joint_list)
                frame_h5.create_dataset("person_to_joint_assoc", data=person_to_joint_assoc)
                t2 = time.time()
                # print messages
                print('{}[{}/{}] process time:{:.3f}s total time:{:.3f}s'.format(
                    time.strftime('%H:%M:%S'), i, length, (t2 - t1), (t2 - t0)))
                if show_visualize_process:
                    cv2.imshow(os.path.basename(output_path), to_plot)
                    cv2.waitKey(1)
        finally:
            out.release()
            out_h5.close()
            cv2.destroyAllWindows()
    print("Prosessed {} items, ignore {} existing items. Saved into {}".format(total_item - ignore_item, ignore_item,
                                                                               output_dir))
    print("All work are Finished！")


if __name__ == "__main__":
    parse = argparse.ArgumentParser(
        "A pre-work for create Action-Datasets, generate .h5 files to "
        "store the joints and skeletons of humans in either videos or pictures")
    parse.add_argument("input", metavar="input_dir", type=str)
    parse.add_argument("output", metavar="output_dir", type=str)
    parse.add_argument("-t", "--input_type", choices=["1to1", "nto1"], default="1to1")
    parse.add_argument('-w', '--weight', type=str, default='./network/weight/pose_model.pth',
                       help='path to the weights file')
    parse.add_argument("-b", "--rebuild", action="store_true", help="rebuild existed file")
    parse.add_argument("-e", "--out_ext", choices=VIDEO_EXT, default=".mp4")
    parse.add_argument("-f", "--resize_factor", type=float, default=1, help="minification factor")
    parse.add_argument("-r", "--frame_ratio", type=int, default=1, help="analyze every [n] frames")
    parse.add_argument("-s", "--process_speed", type=int, default=2,
                       help="Int 1 (fastest, lowest quality) to 4 (slowest, highest quality)")
    parse.add_argument("-l", "--out_length", type=int, default=None,
                       help="frame count for output, default is input length")
    parse.add_argument("-v", "--verb", action="store_true", help="show video canvas")

    args = parse.parse_args()

    main(args)
