import time
from loguru import logger
import os
import cv2


CONFIG= './config.yml'

@logger.catch
def run_on_lego_vid(main_q, cfg):

    time.sleep(10)
    counter = 1
    
    list_buffer_size = int(cfg['VIDEO']['list_buffer_size'])
    vid_dir = cfg['VIDEO']['vid_dir']
    sleep_time = float(cfg['VIDEO']['sleep_time'])

    for filename in os.listdir(vid_dir):
        f = os.path.join(vid_dir, filename)
        logger.info(f'Video Currently {filename}')
    # checking if it is a file
        if os.path.isfile(f):
            vidcap = cv2.VideoCapture(f)
            success,image = vidcap.read()
            
            buffer_list = []
            iter = 0
            while success:   
                buffer_list.append(image)
                time.sleep(sleep_time)
               
                iter += 1
                counter += 1
                
                if iter >= list_buffer_size:
                    
                    main_q.put((buffer_list,time.time()))
                    buffer_list = []
                    iter = 0
                
                success,image = vidcap.read()

