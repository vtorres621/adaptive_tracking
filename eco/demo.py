#Performs YOLO object detection from video captured on webcam and shows detections and bounding box

import cv2
import numpy as np
import sys
#np.set_printoptions(threshold=sys.maxsize)
import vart
import xir
import argparse
import time
import os


from src.utils import *


divider = "---------------------------------------------------"

if __name__ == '__main__':

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()  
    #ap.add_argument('-t', '--threads',      type=int, default=1,                        help='Number of threads. Default is 1')
    ap.add_argument('-m', '--model',        type=str, default='eco.xmodel',             help='Path of xmodel. Default is eco.xmodel')
    #ap.add_argument('-f', '--total_frames', type=int, default=100,                      help='Total frames to capture' )
    #ap.add_argument('-k', '--key_frame',    type=int, default=10,                       help='Keyframes where detection is performed' )

    args = ap.parse_args()
    model = args.model 
    #threads = args.threads 
    #total_frames = args.total_frames
    #key_frame = args.key_frame

    print(divider)
    print('ECO + KF Adaptive subsampling implementation')
    print ('Command line options:')
    #print (' --threads      : ', threads)
    print (' --model        : ', model)
    #print (' --total_frames : ', total_frames)
    #print (' --key_frames   : ', key_frame)
    print(divider)

    
    #Get subgraphs from model
    print("[INFO] Deserializing model subgraphs...")
    g = xir.Graph.deserialize(model)
    subgraphs = get_child_subgraph_dpu(g)
    print("Subgraphs: ", subgraphs)

    #Create DPU Runner
    print("[INFO] Creating DPU Runner...")
    dpu = vart.Runner.create_runner(subgraphs[0], "run")

    #Get DPU info
    #print("[INFO] DPU properties: ")
    inputTensors = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()
    input_ndim = tuple(inputTensors[0].dims)
    output_ndim1 = tuple(outputTensors[0].dims)
    output_ndim2 = tuple(outputTensors[1].dims)
    output_ndim3 = tuple(outputTensors[2].dims)
    output_ndim4 = tuple(outputTensors[3].dims)

    print("Input tensors: ", inputTensors)
    print("Input tensor dimensions: ", input_ndim)
    print("Output tensors: ", outputTensors)
    print("Output tensor 1 dimensions: ", output_ndim1)
    print("Output tensor 2 dimensions: ", output_ndim2)
    print("Output tensor 3 dimensions: ", output_ndim3)
    print("Output tensor 4 dimensions: ", output_ndim4)

    IMAGES_PATH = 'img'
    SAVE_PATH = 'results'
    _, _, files = next(os.walk(IMAGES_PATH))
    files.sort()

    for image in files:
        #Load image
        image_number = image.split('.')
        image_path = os.path.join(IMAGES_PATH,image)
        image = cv2.imread(image_path)

        #Preprocess
        img = [np.array(preprocessing(image,(448,448)), dtype=np.float32)]
        
        #Prepare input and output data
        inputData = [np.empty(input_ndim, dtype=np.float32, order="C")]
        outputData = [np.empty(output_ndim1, dtype=np.float32, order="C"),
                      np.empty(output_ndim2, dtype=np.float32, order="C"),
                      np.empty(output_ndim3, dtype=np.float32, order="C"),
                      np.empty(output_ndim4, dtype=np.float32, order="C")]
        #outputData = []

        #Load image into inputData
        imageRun = inputData[0]
        imageRun[0, ...] = img[0].reshape(input_ndim[1:])

        #Show input image
        #cv2.imshow("Result", inputData[0][0])
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        
        #Run DPU
        job_id = dpu.execute_async(inputData[0],outputData)
        dpu.wait(job_id)

        #Print Output
        feature1 = outputData[0][0]
        feature2 = outputData[1][0]
        feature3 = outputData[2][0]
        feature4 = outputData[3][0]
        #print("Feature 1 shape: ", feature1.shape)
        #print("Feature 1 data: ", feature1)
        #print("Feature 2 shape: ", feature2.shape)
        #print("Feature 2 data: ", feature1)
        #print("Feature 3 shape: ", feature3.shape)
        #print("Feature 3 data: ", feature1)
        #print("Feature 4 shape: ", feature4.shape)
        #print("Feature 4 data: ", feature1)

        #Save array
        print('Saving... ', image_number[0])
        #save_path = os.path.join(SAVE_PATH,image_number[0]+'layer1')
        #np.save(save_path, feature1)
        #save_path = os.path.join(SAVE_PATH,image_number[0]+'layer2')
        #np.save(save_path, feature2)
        #save_path = os.path.join(SAVE_PATH,image_number[0]+'layer3')
        #np.save(save_path, feature3)
        #save_path = os.path.join(SAVE_PATH,image_number[0]+'layer4')
        #np.save(save_path, feature4) 
