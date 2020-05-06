import os
import re
import sys
import math
import argparse
from collections import OrderedDict

import numpy as np

from data_loaders import data_loaders, datasets, pre_processing
from parse_config import init_all_helper
from utils import read_json, write_json


def get_save_path(save_path, idx):
    """Get save path for a part of a dataset"""
    if idx == 0:
        return save_path
    filename, fileext = os.path.splitext(save_path)
    filename = filename + "_{}".format(idx) + fileext
    return filename


def save_part(images, info, out_path, part_idx=0, verbose=True):
    # Save
    save_path = get_save_path(out_path, idx=part_idx)
    np.save(save_path, images)
    # Cache useful information
    info["parts"][part_idx] = OrderedDict(
        save_path=save_path, shape=list(images.shape))
    if verbose:
        print("Successfully saved part {} consisting of {} images "
              "to {}.".format(part_idx, len(images), save_path))


def extract_data(data_loader, out_path, auto_split=True, num_parts=1,
                 max_mem_fraction=None, verbose=True):
    if verbose:
        from tqdm import tqdm
        log = print
    else:
        tqdm = log = lambda x: x  # dummy identity function
    # Parameters
    images = list()
    info = OrderedDict()  # store information such as len, dtype, ...
    batch_size = data_loader.batch_size
    num_batches = len(data_loader)
    num_batches_in_a_part = None if auto_split else num_batches
    part_idx = -1

    for batch_idx, batch in enumerate(tqdm(data_loader)):
        batch = batch.cpu().numpy()
        # Auto split
        if num_batches_in_a_part is None:
            # Equally split into several parts
            if num_parts is not None:
                num_batches_in_a_part = math.ceil(num_batches / num_parts)
            # Split into parts based on the amount of available memory
            else:
                import psutil
                avai_mem = psutil.virtual_memory().available
                batch_mem = batch.nbytes  # batch memory in bytes
                # Cap at (max_mem_fraction * 100)% memory usage
                num_batches_in_a_part = (int(max_mem_fraction * avai_mem)
                                         // batch_mem)
                num_parts = math.ceil(num_batches / num_batches_in_a_part)
            if num_parts > 1 and verbose:
                print("Splitting dataset into {} parts, each of which consists"
                      " of {} batches ({} images)".format(
                        num_parts, num_batches_in_a_part,
                        num_batches_in_a_part * batch_size))
        # Cache useful information
        if "dtype" not in info:
            info["dtype"] = str(batch.dtype)
            info["num_parts"] = num_parts
            info["num_images"] = len(data_loader.dataset)
            info["parts"] = OrderedDict()
        # Start of a new part
        if batch_idx % num_batches_in_a_part == 0:
            # Save previous part
            if batch_idx != 0:
                part_idx = batch_idx // num_batches_in_a_part - 1
                save_part(images, info, out_path=out_path, part_idx=part_idx,
                          verbose=verbose)
            # Allocate empty array for this part
            del images
            images = np.empty(
                shape=[batch_size * num_batches_in_a_part, *batch.shape[1:]],
                dtype=batch.dtype)
        # Append batch
        start = (batch_idx % num_batches_in_a_part) * batch_size
        end = start + len(batch)
        images[start:end] = batch
    # For the last part
    images = images[:end]
    part_idx += 1
    save_part(
        images, info, out_path=out_path, part_idx=part_idx, verbose=verbose)
    # Save `info` to a json file
    curr_dir = os.path.split(out_path)[0]
    info_path = os.path.join(curr_dir, "info.json")
    write_json(info, info_path)


def main(args):
    if args.auto_split and args.num_parts is None \
            and args.max_mem_fraction is None:
        raise ValueError('One of flags ["num_parts", "max_mem_fraction"] must '
                         'be set when "auto_split" is set to True.')
    # Read json
    config = read_json(args.config)
    # Initialize dataset
    type_mapping = {
        "data_loader": data_loaders,
        "dataset": datasets,
        "pre_processing": pre_processing
    }
    init_all_helper(config, type_mapping)
    # Get DataLoader object
    data_loader = config["data_loader"]["obj"]
    # Extract data
    extract_data(
        data_loader, out_path=args.out_path, auto_split=args.auto_split,
        num_parts=args.num_parts, max_mem_fraction=args.max_mem_fraction,
        verbose=args.verbose)


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        description="Load, transform and extract data to a *.npy file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-c", "--config", default=None, type=str,
        help="Config file path")
    parser.add_argument(
        "-o", "--out_path", default=None, type=str,
        help="Path to write the output *.npy file")
    parser.add_argument(
        "-v", "--verbose", default=False, action="store_true",
        help="If True, print out the data loading progress and other useful "
             "debugging information. This requires tqdm to be installed.")
    parser.add_argument(
        "-a", "--auto_split", default=False, action="store_true",
        help="If True, auto split dataset into several parts and one of "
             "['max_mem_fraction', 'num_parts'] must be specified.")
    parser.add_argument(
        "-n", "--num_parts", default=None, type=int,
        help="If `auto_split` is set, split dataset into `num_parts` parts. "
             "This option is preferred over `max_mem_fraction`.")
    parser.add_argument(
        "-m", "--max_mem_fraction", default=None, type=float,
        help="If `auto_split` is set, split dataset into several parts such "
             "that each part does not exceeed a maximum fraction of memory. "
             "This requires psutil to be installed.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
