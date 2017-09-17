"""Download images from zalando.se.

item_image_url_dict consists of dictionary of dictionary.
"""
import argparse
import json
import multiprocessing
import os
import subprocess

import tqdm.tqdm as tqdm

from download_images import DataDownloader
import zalando_data


def main():
    # TODO(crcrpar): define dataset_json generator
    paraser = argparse.ArgumentParser(description='Dataset Generator')
    paraser.add_argument('--root', default='data')
    paraser.add_argument('--base_root', default='images')
    paraser.add_argument('--update', default=True,
                         help='Update images or not')
    paraser.add_argument('--limit', default=15000, type=int,
                         help='limit of the number of articles')
    args = paraser.parse_args()

    image_root = os.path.join(args.root, args.base_root)
    data_downloader = DataDownloader(root=image_root,
                                     update=args.update,
                                     limit=args.limit)


if __name__ == '__main__':
    main()
