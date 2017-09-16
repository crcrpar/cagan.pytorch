import argparse
from io import BytesIO
import os
from PIL import Image

import tqdm.tqdm as tqdm

import zalando_data


def download_images(article_id_list, image_root):
    url_base = 'https://api.zalando.com/articles/{}/media'
    for article_id in tqdm(article_id_list):
        media_url = url_base.format(article_id)


def main():
    paraser = argparse.ArgumentParser(description='Dataset Generator')
    paraser.add_argument('--root', default='data')
    paraser.add_argument('--base_root', default='images')
    args = paraser.parse_args()
    image_root = os.path.join(args.root, args.base_root)
    if not os.path.isdir(image_root):
        os.makedirs(image_root)
    pass


if __name__ == '__main__':
    main()
