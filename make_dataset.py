"""Download images from zalando.se.

item_image_url_dict consists of dictionary of dictionary.
"""
import argparse
import json
import multiprocessing
import os
import subprocess

import tqdm.tqdm as tqdm

import zalando_data


CMD_BASE = 'wget -q -t 3 -nc {url} -O {dst}'


def work(cmd):
    pid = subprocess.Popen(cmd).pid
    return pid


def get_tops_categories(all_categories=None):
    _cached = 'zalando_all_categories.txt'
    if all_categories is None:
        if not os.path.exists(_cached):
            all_categories = zalando_data.get_possible_category_keys()
            with open(_cached, 'w') as f:
                f.write('\n'.join(all_categories))
        else:
            with open(_cached) as f:
                print('\tload cached file')
                all_categories = [line.strip('\n') for line in f]

    tops_categories = list()
    for category in all_categories:
        if 'shirt' in category:
            tops_categories.append(category)
        if 'pullover' in category or 'pullovers' in category:
            tops_categories.append(category)
        if 'hoody' in category or 'hoodies' in category:
            tops_categories.append(category)
        if 'sweater' in category:
            tops_categories.append(category)
        if 'jacket' in category:
            tops_categories.append(category)
        if 'tunic' in category:
            tops_categories.append(category)
        if 'blouse' in category:
            tops_categories.append(category)
        if 'dress' in category:
            tops_categories.append(category)

    return tops_categories


def get_tops_model_item_image_urls(tops_categories, out_path=None):
    from datetime import datetime as dt
    if out_path is None:
        out_path = 'tops_images_{}.txt'.format(dt.now().strftime('%m%d_%H%M'))
    item_image_url_dict = {}
    query_builder = zalando_data.Query()
    query_builder.set_categories(tops_categories)
    counter = 0
    for i, item in tqdm(enumerate(zalando_data.get_articles_all_pages(query_builder))):
        if counter >= 15000:
            break
        images = item['media']['images']
        type_set = [i['type'] == 'MODEL' for i in images]
        if not any(type_set):
            continue
        tmp_table = {}
        for image_table in images:
            counter += 1
            if image_table['type'] == 'MODEL':
                tmp_table[image_table['type']] = image_table['smallHdUrl']
            elif image_table['type'] in ('NON_MODEL', 'MANUFACTURER'):
                tmp_table[image_table['type']] = image_table['smallHdUrl']
            elif image_table['type'] == 'UNSPECIFIED':
                tmp_table[image_table['type']] = image_table['smallHdUrl']
            elif image_table['type'] == 'STYLE':
                tmp_table[image_table['type']] = image_table['smallHdUrl']
            else:
                pass
            item_image_url_dict[item['id']] = tmp_table

            line = ''
            for key, value in tmp_table.items():
                line += '{}:{}\t'.format(key, value)

            with open(out_path, 'a') as f:
                f.write(line + '\n')
    return item_image_url_dict


def prepare_job_list(item_image_url_dict, base_root):
    job_list = list()
    for dir_name, item_dict in item_image_url_dict.items():
        root = os.path.join(base_root, dir_name)
        if not os.path.isdir(root):
            os.makedirs(root)
        for key, url in item_dict.items():
            dst = '{}.jpg'.format(key.lower())
            job_list.append(CMD_BASE.format(url=url, dst=dst))
    return job_list


def get_images(item_image_url_dict, base_root):
    job_list = prepare_job_list(item_image_url_dict, base_root)
    n_cpu = multiprocessing.cpu_count() - 1
    pool = multiprocessing.Pool(processes=n_cpu)
    results = list()
    r = pool.map_async(work, job_list, callback=results.append)
    r.wait()
    pool.close()
    pool.join()
    return results


def main():
    paraser = argparse.ArgumentParser(description='Dataset Generator')
    paraser.add_argument('--root', default='data')
    paraser.add_argument('--base_root', default='images')
    paraser.add_argument('--item_image_url_dict', default=None,
                         help='path to dictionary')
    args = paraser.parse_args()
    if args.item_image_url_dict is None:
        tops_categories = get_tops_categories()
        item_image_url_dict = get_tops_model_item_image_urls(
            tops_categories, args.root)
    else:
        item_image_url_dict = json.load(open(args.item_image_url_dict))
    image_root = os.path.join(args.root, args.base_root)
    if not os.path.isdir(image_root):
        os.makedirs(image_root)

    results = get_images(item_image_url_dict, image_root)


if __name__ == '__main__':
    main()
