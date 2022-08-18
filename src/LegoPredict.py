import cv2 as cv
import numpy as np
from loguru import logger
from datetime import datetime
import tensorflow as tf
import sys

class LegoPrediction:

    def __init__(self,cfg):
        self.batch_size = cfg['VIDEO']['list_buffer_size'] # batch of 20 frames
        self.width = cfg['VIDEO']['vid_width']
        self.height = cfg['VIDEO']['vid_height']
        
    def load_model(self):
        self.new_model = tf.keras.models.load_model('.\models')#('C:\\Users\elliscll\Documents\Lego_Project\src\models')
        # Check its architecture
        print(self.new_model.summary())

        import csv
        self.labels = {}
        with open('labels.csv', mode='r') as inp:
            reader = csv.reader(inp)
            self.labels = {rows[0]:rows[1] for rows in reader}
        logger.debug("model and labels loaded")

    def make_prediction(self,itm_list):
        
            try:
                    pred_time_start = datetime.now()
                    itm_list_new = []
                    i = 0

                    for img in itm_list[::int(self.batch_size)]:

                        if(cv.countNonZero(cv.cvtColor(img, cv.COLOR_BGR2GRAY)) != 0): #if blank image, do not perform prediction                   
                            itm_list_new.append(
                                cv.resize(
                                    img
                                ,(self.width, self.height) #(width,height) for cv
                                )
                            )
                    
                    itm_list_new = np.array(itm_list_new)

                    if np.any(itm_list_new) :
                        pred_prob = self.new_model.predict(itm_list_new) # make prediction
                        pred_classes = np.argmax(pred_prob, axis=1)
                        pred_label = [self.labels[str(k)] for k in pred_classes]
                        pred_time_end = datetime.now()
                        logger.debug(f'pred time: {pred_time_end - pred_time_start}')

                    else:
                        pred_classes = "NA"
                        pred_label = "blank image"

                    logger.debug(f'pred_classes: {pred_classes}, pred_label: {pred_label}')
                
            except Exception as e:
                logger.exception(f'Error in Prediction thread: {e}')
                sys.exit()
