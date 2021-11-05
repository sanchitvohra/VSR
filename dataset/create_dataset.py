import torch
from tfrecord.torch.dataset import TFRecordDataset

import argparse
from pytube import extract
import os

import numpy as np
import tensorflow as tf

import pickle

import asyncio
import aiohttp

parser = argparse.ArgumentParser(description='Create dataset from partitions of Youtube-8M')
parser.add_argument('--dataset', help='Path to dataset directory containing tfrecord files')

args = parser.parse_args()
dataset_path = args.dataset

if not os.path.isdir(dataset_path):
    print("Please enter a valid path")
    exit(1)

### check existence of train/test/validate folders + make them
# sub_dirs = ["train", "test", "validate"]

# for dir in sub_dirs:
#     full_path = os.path.join(dataset_path, dir)
#     if not os.path.isdir(full_path):
#         os.mkdir(full_path)

async def get_video_link(session, url):
    async with session.get(url) as resp:
        if resp.status == 200:
            video_id = await resp.text()
            hash = video_id.split(',')[1][1:-3]
            link = "https://www.youtube.com/watch?v=" + hash
            return link

async def get_video_data(session, url):
    async with session.get(url) as resp:
        if resp.status == 200:
            yt_html = await resp.text()
            status, messages = extract.playability_status(yt_html)
            for reason in messages:
                if status == 'UNPLAYABLE':
                    return None
                elif status == 'LOGIN_REQUIRED':
                    return None
                elif status == 'ERROR':
                    return None
                elif status == 'LIVE_STREAM':
                    return None
            yt_strm = extract.get_ytplayer_config(yt_html)
            if 'streamingData' in yt_strm:
                if 'formats' in yt_strm['streamingData']:
                    for stream in yt_strm['streamingData']['formats']:
                        if 'mimeType' in stream and stream['mimeType'].find('video/mp4') >= 0 and 'qualityLabel' in stream and stream['qualityLabel'] == '1080p':
                            return url
                if 'adaptiveFormats' in yt_strm['streamingData']:
                    for stream in yt_strm['streamingData']['adaptiveFormats']:
                        if 'mimeType' in stream and stream['mimeType'].find('video/mp4') >= 0 and 'qualityLabel' in stream and stream['qualityLabel'] == '1080p':
                            return url
        
async def parse_tfrecord(urls, file):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for url in urls:
            tasks.append(asyncio.ensure_future(get_video_link(session, url)))

        video_urls = await asyncio.gather(*tasks)
        tasks = []
        for video_url in video_urls:
            if video_url:
                tasks.append(asyncio.ensure_future(get_video_data(session, video_url)))

        video_1080 = await asyncio.gather(*tasks)
        res = list(filter(None, video_1080))
        return res

async def parse_directory():
    while(1):
        try:
            for file in os.listdir(dataset_path):
                if file.find(".tfrecord") >= 0 and file.find("train") >= 0:
                    name = file.split('.')[0]
                    if not os.path.exists(os.path.join(dataset_path, name + '.pkl')):
                        all_urls = []
                        tfrecord_path = os.path.join(dataset_path, file)
                        index_path = None
                        description = {"id": "byte", "mean_rgb": "float", "mean_audio": "float"}
                        dataset = TFRecordDataset(tfrecord_path, index_path, description)
                        loader = torch.utils.data.DataLoader(dataset, batch_size=1)

                        urls = []
                        for data in loader:
                            id = bytes(data['id'][0].numpy().tolist()).decode('utf-8')
                            url = 'http://data.yt8m.org/2/j/i/' + id[:2] + '/' + id + '.js'
                            urls.append(url)
                            
                            if len(urls) == 100:           
                                video_urls = await parse_tfrecord(urls, file)
                                all_urls.append(video_urls)
                                urls = []

                        if len(urls) > 0:
                            video_urls = await parse_tfrecord(urls, file)
                            all_urls.append(video_urls)
                            with open(os.path.join(dataset_path, name + '.pkl'), 'wb') as handle:
                                print("writing ", name)
                                pickle.dump(all_urls, handle)
            break
        except:
            pass
        
asyncio.run(parse_directory())