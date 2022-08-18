import multiprocessing
from src.LegoPredict import LegoPrediction
import time
from loguru import logger
from src.Save_Edges import save_edges_vid
import queue
import traceback

@logger.catch
def __prediction_run(cfg, q_in):

    pred = LegoPrediction(cfg)
    pred.load_model()

    time.sleep(1)

    while 1:

        try:

            itm_list, timer, work_id  = q_in.get()
            output = pred.make_prediction(itm_list)     
            logger.info(f"Work_id: {work_id} | Timer: {timer} | Time elapsed: {round(time.time()-timer,4)} seconds | Prediction output calculated")

        except Exception as e:
            logger.exception(f'Error in prediction thread loop: {e}')

@logger.catch
def __init_helper(cfg):
    queue_list = []
    process_list = []

    if cfg['THREADS']['Prediction']:
        autoencoder_queue = multiprocessing.Queue()
        queue_list.append(autoencoder_queue)
        pautoencoder = multiprocessing.Process(target=__prediction_run, args=(cfg,autoencoder_queue,), name = 'Model prediction') 
        pautoencoder.daemon = True
        pautoencoder.start()
        process_list.append(pautoencoder)
        logger.info("Autoencoder thread started at PID: {}", pautoencoder.pid)

    if cfg['THREADS']['EdgesSave']:
        video_save = multiprocessing.Queue()
        queue_list.append(video_save)
        pvideo_save = multiprocessing.Process(target=save_edges_vid, args=( cfg,video_save,), name = 'video save')
        pvideo_save.daemon = False
        pvideo_save.start()
        process_list.append(pvideo_save)
        logger.info(f"Video save thread started at PID: {pvideo_save.pid}")

    return queue_list,process_list


@logger.catch
def program_run(cfg,video_queue,error_queue):

    queue_list,process_list = __init_helper(cfg)
    work_id = 0

    while 1:
        
        for proc in process_list:
            print(f"Process details for {proc.name}: pid={proc.pid}, exitcode={proc.exitcode}, is_alive={proc.is_alive()}")
        try:
            exc_flag = error_queue.get(timeout=0.5)
            if exc_flag:
                logger.critical("EXCEPTION_FLAG = True")
                return process_list
        except queue.Empty:
            pass
        
        try:
            work_id = work_id % 100
            
            itm,timer = video_queue.get(timeout=60)
            
            logger.info(f"Work_id: {work_id} | Timer: {timer} | Time elapsed: {round(time.time()-timer,4)} seconds | Batch of frames of size {len(itm)} obtained from video queue")

            for q in queue_list:
                q.put((itm,timer,work_id))

            work_id +=1
        except Exception as e:
            logger.warning(f"Nothing to process for metrics thread. Exception - {e}")
            logger.debug(traceback.format_exc())
            return process_list
    