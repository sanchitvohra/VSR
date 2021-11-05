import asyncio
import argparse
import os
import pickle

parser = argparse.ArgumentParser(description='Create dataset from partitions of Youtube-8M')
parser.add_argument('--dataset', help='Path to dataset directory containing tfrecord files')
parser.add_argument('--partition', help='Partition No')

args = parser.parse_args()
dataset_path = args.dataset

if not os.path.isdir(dataset_path):
    print("Please enter a valid path")
    exit(1)

full_path = os.path.join(dataset_path, "videos")
if not os.path.isdir(full_path):
    os.mkdir(full_path)


curdir = os.getcwd()
script_path = os.path.join(curdir, "postprocessing.py")

async def download_command(video_url):
    process = await asyncio.create_subprocess_exec('python3', script_path, video_url, full_path,
    stdout=asyncio.subprocess.DEVNULL,
    stderr=asyncio.subprocess.DEVNULL)
    stdout, stderr = await process.communicate()
    print(video_url, stderr)

parition_path = os.path.join(dataset_path, "partition"+args.partition)

async def main():
    for file in os.listdir(parition_path):
        if file.find(".pkl") >= 0:
            name = file.split('.')[0]
            urls = []
            with open(os.path.join(parition_path, file), 'rb') as handle:
                urls = pickle.load(handle)
            tasks = []
            for sublist in urls:
                for url in sublist:
                    id = url[url.find("?v=")+3:]
                    if not os.path.exists(os.path.join(full_path, id + "_0.mp4")):
                        tasks.append(asyncio.ensure_future(download_command(url)))
                    if len(tasks) >= 20:
                        await asyncio.gather(*tasks)
                        tasks = []
            if len(tasks) > 0:
                await asyncio.gather(*tasks)
                tasks = []

asyncio.run(main())

