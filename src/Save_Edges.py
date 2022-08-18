#In this case, we have two scenes we're concerned about - 'blank' and 'lego brick present'
#Because our scenes are very simple, we can use a super simple form of scene detection using only Edge Detection

import cv2 as cv
import time
import cv2
import time
from datetime import datetime
from  multiprocessing import Process
from loguru import logger

def find_edges(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(gray_img, (3,3), 0)
    edge_img = cv2.Canny(img_blur,100,200)
    edge_img = cv2.cvtColor(edge_img, cv2.COLOR_GRAY2BGR)
    return gray_img, img_blur, edge_img

def save_edges_vid(cfg,q_in):
    
    save_dir = str(cfg['VIDEO_SAVE']['save_dir'])
    frame_size = int(cfg['VIDEO_SAVE']['video_frame_size'])
    time.sleep(7)

    itm_list, timer, _ = q_in.get()
    vid_name = f'{datetime.fromtimestamp(timer).strftime("%m_%d_%y_%H_%M_%S_%f")}.mp4'
    vid_path = save_dir+vid_name
    print(vid_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(vid_path, fourcc, 20, (400,400))
    frame_iter = 0

    while 1:
        try:

            for img in itm_list:
                _,_,edge_img = find_edges(img)
                out.write(edge_img)
                frame_iter +=1

            if frame_iter > frame_size:
                out.release()
                vid_name = f'{datetime.fromtimestamp(timer).strftime("%m_%d_%y_%H_%M_%S_%f")}.mp4'
                vid_path = save_dir + vid_name
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(vid_path, fourcc, 20, (400,400))
                frame_iter = 0

        except Exception as e:
            
            frame_iter = 0
            logger.exception("Error in save video thread: {}", e)
            out.release()
            vid_name = f'{datetime.fromtimestamp(timer).strftime("%m_%d_%y_%H_%M_%S_%f")}.mp4'
            vid_path = save_dir + vid_name
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(vid_path, fourcc, 20, (400,400))
        itm_list, timer, _ = q_in.get()
