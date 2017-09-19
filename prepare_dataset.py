import argparse
from datetime import datetime as dt
import json
from logging import DEBUG
from logging import getLogger
from logging import StreamHandler
import multiprocessing
import os
import shutil
import subprocess

import zalando_data as zalando


class DataDownloader(object):

    CMD_BASE = 'wget -q -t 3 -nc {url} -O {dst}'
    _cached_all_categories = 'data/zalando_all_categories.txt'
    _cached_tops_categories = 'data/zalando_tops_categories.txt'
    _cached_item_image_url_dict = 'data/item_image_url_dict.json'
    _cached_prev_item_image_url_dict = 'data/prev_item_image_url_dict.json'
    _cached_job_list = 'data/job_list_{}.txt'

    def __init__(self, root, update=True, limit=15000):
        self.root = root
        if not os.path.isdir(self.root):
            os.makedirs(self.root)
        self.update = update
        self.limit = limit
        # set logger
        logger = getLogger(name='Data Downloader')
        handler = StreamHandler()
        handler.setLevel(DEBUG)
        logger.setLevel(DEBUG)
        logger.addHandler(handler)
        self.logger = logger
        self.logger.debug('Instance is initialized.')

    def run(self):

        def work(cmd):
            pid = subprocess.Popen(cmd.split(' ')).pid
            return pid

        start = dt.now().strftime('%m/%d, %H:%M:%S')
        self.logger.debug('Started to run at {}'.format(start))
        self.logger.debug('[Download Images] preparing list of DL jobs...')
        job_list = self.prepare_job_list()
        self.logger.debug(
            '[Download Images] # of jobs is {}'.format(len(job_list)))
        results = list()
        n_cpu = multiprocessing.cpu_count() - 1
        self.logger.debug('[Download Images] run {} processes'.format(n_cpu))
        start = dt.now()
        pool = multiprocessing.Pool(processes=n_cpu)
        r = pool.map_async(work, job_list, callback=results.append)
        r.wait()
        pool.close()
        pool.join()
        end = dt.now()
        duration = (end - start).total_seconds() / 60.
        self.logger.debug('[Download Images] finished jobs')
        self.logger.debug('Jobs from {} to {}, took {} minutes'.format(
            start.strftime('%m/%d, %H:%M:%S'),
            end.strftime('%m/%d, %H:%M:%S'),
            duration))
        self.logger.debug('[Make triplet]')
        # self.make_triplet_list()

    def make_triplet_list(self):
        item_image_url_dict = self.get_tops_model_item_image_urls(
            _update=False, save=False)
        pass

    def prepare_job_list(self):

        def make_job_from_article_dict(*args, **kwds):
            article_id = args[0]
            base_root = os.path.join(self.root, article_id)
            tmp_job_list = list()
            if not os.path.isdir(base_root):
                os.makedirs(base_root)
            for idx, (image_type, url) in enumerate(kwds.items()):
                dst = os.path.join(
                    base_root, '{}_{}.jpg'.format(image_type, idx))
                job = DataDownloader.CMD_BASE.format(url=url, dst=dst)
                tmp_job_list.append(job.split(' '))
            return tmp_job_list

        # load item_image_url_dict
        if os.path.exists(DataDownloader._cached_item_image_url_dict):
            with open(DataDownloader._cached_item_image_url_dict) as f:
                item_image_url_dict = json.load(f)
        else:
            item_image_url_dict = self.get_tops_model_item_image_urls()

        # run preparing a list of jobs
        job_list = list()
        n_cpu = multiprocessing.cpu_count() - 1
        p = multiprocessing.Pool(n_cpu)
        for item_id, type_url_dict in item_image_url_dict.items():
            r = p.apply_async(make_job_from_article_dict,
                              args=(item_id,), kwds=type_url_dict,
                              callback=job_list.extend)
            r.wait()
        p.close()
        p.join()

        ts = dt.now().strftime('%m%d_%H%M')
        with open(DataDownloader._cached_job_list.format(ts), 'w') as f:
            f.write('\n'.join(job_list))

        return job_list

    def get_tops_model_item_image_urls(self, _update=False, save=True):

        def parse_article(article):
            article_id = article['id']
            media_images = article['media']['images']
            is_NON_MODEL = [i['type'] == 'NON_MODEL' for i in media_images]
            if all(is_NON_MODEL):
                return {article_id: None}

            tmp_dict = dict()
            for idx, image in enumerate(media_images):
                _key = '{}_{}'.format(image['type'], idx)
                tmp_dict[_key] = image['smallHdUrl']
            return {article_id: tmp_dict}

        if os.path.exists(DataDownloader._cached_tops_categories):
            with open(DataDownloader._cached_tops_categories) as f:
                tops_categories = [line.strip('\n') for line in f]
        else:
            tops_categories = self._get_tops_categories()

        raw_itemid_type_url_dict = dict()
        query_builder = zalando.Query()
        query_builder.set_categories(tops_categories)
        n_cpu = multiprocessing.cpu_count() - 1
        pool = multiprocessing.Pool(processes=n_cpu)
        r = pool.map_async(parse_article,
                           zalando.get_articles_all_pages(query_builder),
                           callback=raw_itemid_type_url_dict.update)
        r.wait()
        pool.close()
        pool.join()

        item_image_url_dict = dict()
        for article_id, dictionary in raw_itemid_type_url_dict.item():
            if isinstance(dictionary, dict):
                item_image_url_dict[article_id] = dictionary

        if self.update:
            self._copy_prev_dict()
        if save:
            with open(DataDownloader._cached_item_image_url_dict, 'w') as f:
                json.dump(item_image_url_dict, f)

        return item_image_url_dict

    def _copy_prev_dict(self):
        if os.path.exists(DataDownloader._cached_prev_item_image_url_dict):
            os.remove(DataDownloader._cached_prev_item_image_url_dict)
        shutil.copy(src=DataDownloader._cached_item_image_url_dict,
                    dst=DataDownloader._cached_prev_item_image_url_dict)

    def _get_tops_categories(self):
        if os.path.exists(DataDownloader._cached_all_categories):
            with open(DataDownloader._cached_all_categories) as f:
                all_categories = [line.strip('\n') for line in f]
        else:
            all_categories = zalando.get_possible_category_keys()
            with open(DataDownloader._cached_all_categories, 'w') as f:
                f.write('\n'.join(all_categories))

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

        with open(DataDownloader._cached_tops_categories, 'w') as f:
            f.write('\n'.join(tops_categories))
        return tops_categories


def main():
    paraser = argparse.ArgumentParser(description='Download Images')
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
    data_downloader.run()


if __name__ == '__main__':
    main()
