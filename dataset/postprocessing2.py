from pytube import YouTube
import os
import sys
import subprocess
import random
import cv2
import sys

import requests

NUM_SNIPPETS = 10

video_url = sys.argv[1]
video_dataset_path = sys.argv[2]

id = video_url[video_url.find("?v=")+3:]

raw_dir_path = os.path.join(os.getcwd(), "raw")
raw_video_path = os.path.join(raw_dir_path, id + "_raw.mp4")

yt = YouTube(video_url)

streams = yt.streams.filter(file_extension="mp4", res="1080p")

if len(streams) == 0:
    exit()

stream = streams[0]
stream.download(output_path = raw_dir_path, filename = id + "_raw.mp4")

cap = cv2.VideoCapture(raw_video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
cap.release()
duration = str(format(30/fps, '.3f')).zfill(6)

for snip in range(NUM_SNIPPETS):
    start_frame = random.randint(45, frames-45)
    start_time = start_frame / fps
    milliseconds = str(format(start_time % 1, '.3f'))
    seconds = int(start_frame / fps)
    minutes = int(seconds / 60)
    seconds = seconds - 60 * minutes
    fmt = str(minutes).zfill(2) + ':' + str(seconds).zfill(2) + milliseconds[1:]
    output_path = os.path.join(video_dataset_path, id + "_"+str(snip)+".mp4")
    ffmpeg = subprocess.Popen(["ffmpeg", "-y", "-ss", "00:" + fmt, "-i", raw_video_path, "-t", "00:00:" + duration, "-c", "copy", output_path]
    ,stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    stdout, stderr = ffmpeg.communicate()

if os.path.exists(raw_video_path):
    os.remove(raw_video_path)