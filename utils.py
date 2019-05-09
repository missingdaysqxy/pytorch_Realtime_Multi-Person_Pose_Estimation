# -*- coding: utf-8 -*-
# @Time    : 2019/5/8 19:48
# @Author  : liuqixuan
# @Email   : qxsoftware@163.com
# @File    : qxutils.py
# @Software: PyCharm

import os
import sys
from warnings import warn


# process bar
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


# Organize I/O Paths
def organize_1to1_io_paths(input_dir, input_exts, output_dir, output_ext):
    # type:(str,list,str,str)->dict
    if not os.path.exists(input_dir):
        raise FileNotFoundError("File not exist in {}".format(input_dir))
    io_paths = {"input": [], "output": [], "type": "1to1"}
    if os.path.isdir(input_dir):
        for root, dirs, files in os.walk(input_dir):
            rel_path = os.path.relpath(root, input_dir)
            for file in files:
                name, ext = os.path.splitext(file)
                if ext.lower() in input_exts:
                    input_path = os.path.join(root, file)
                    output_path = os.path.join(output_dir, rel_path, name + output_ext)
                    io_paths["input"].append(input_path)
                    io_paths["output"].append(output_path)
                else:
                    warn("Unsupported format: %s" % file)
    else:
        name, ext = os.path.splitext(input_dir)
        assert ext.lower() in input_exts, "Unsupported format: %s" % input_dir
        output_path = os.path.join(output_dir, os.path.basename(name) + output_ext)
        io_paths["input"].append(input_dir)
        io_paths["output"].append(output_path)
    return io_paths


def organize_Nto1_io_paths(input_dir, input_exts, output_dir, output_ext):
    # type:(str,list,str,str)->dict
    if not os.path.exists(input_dir):
        raise FileNotFoundError("File not exist in {}".format(input_dir))
    io_paths = {"input": [], "output": [], "type": "nto1"}
    if os.path.isdir(input_dir):
        for root, dirs, files in os.walk(input_dir):
            rel_path = os.path.relpath(root, input_dir)
            image_list = []
            for file in files:
                name, ext = os.path.splitext(file)
                if ext.lower() in input_exts:
                    image_path = os.path.join(root, file)
                    image_list.append(image_path)
                else:
                    warn("Unsupported format: %s" % file)
            if len(image_list) > 0:
                output_path = os.path.join(output_dir, rel_path + output_ext)
                image_list = sorted(image_list)
                io_paths["input"].append(image_list)
                io_paths["output"].append(output_path)
    else:
        name, ext = os.path.splitext(input_dir)
        assert ext.lower() in input_exts, "Unsupported format: %s" % input_dir
        output_path = os.path.join(output_dir, os.path.basename(name) + output_ext)
        io_paths["input"].append([input_dir])
        io_paths["output"].append(output_path)
    return io_paths
