#!/usr/bin/python3

import multiprocessing
from multiprocessing import Process, Queue, Manager
import yaml
from src import WorkThread
import src.CaptureVid as CaptureVid
import time
from loguru import logger
import sys

time.sleep((1))

CONFIG='./config.yml'

LOGFILE = './logs/output_{time}.log' 
logger.add(LOGFILE, enqueue=True, backtrace=True, rotation='1 week', retention="10 days", compression="zip") # create new log file after every week and compress it to zip

video_queue=Queue()
error_queue = Queue()

# will be called in case of any exception to terminate all active processes
def myexcepthook(exctype, value, traceback):
    for p in multiprocessing.active_children():
       print(p) 
       p.terminate()

cfg= None
lock = None

if __name__ == '__main__':

    try:
        cfg = None
        with open(CONFIG, 'r') as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

        m = Manager()
        lock = m.Lock()

        sys.excepthook = myexcepthook
        
        logger.info("Starting up threads...")
        
        p0 = Process(target=CaptureVid.run_on_lego_vid, args=(video_queue, cfg))
        p0.daemon = True
        p0.start()
        logger.info("Test Video thread started at PID: {}", p0.pid)
       
        process_list = WorkThread.program_run(cfg,video_queue,error_queue)

        for p in ([p0]+process_list):
            print(f"Terminating process {p.pid}")
            p.terminate()
        sys.exit()

    except Exception as ex:
       logger.info(f"Catching exception in main program - {ex}")
       sys.exit()
