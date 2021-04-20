from Network import Inference
import argparse
import os
import sys
import numpy as np
import cv2
import logging as lg



def build_argparser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--m','--model',type=str,help='path to your xml file'
                        ,required=True)
    parser.add_argument('--i','--input_data',type=str,
                        default='CAM',required=True,
                        help='Path for your input data image/video, if no path entered the camera will be used')
    parser.add_argument('--pt','--pro_threshold',type=float,help='probability threshold',default=0.9)
    parser.add_argument('--d','--device',type=str,default='CPU',help='inference device for the model, default is CPU')
    args = parser.parse_args()

    return args


def draw_bbox(image,model_output,threshold):
    # extract image shape
    h,w,_ = image.shape
    # process model output
    for box in model_output[0][0]:
        conf = box[2]
        if conf > threshold:
            xmin = int(w * box[3])
            ymin = int(h * box[4])
            xmax = int(w * box[5])
            ymax = int(h * box[6])

            image = cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(0,0,255),2)
            image = cv2.putText(image,str(round(conf,2)),(xmin+5,ymin-15),fontFace=cv2.FONT_HERSHEY_SIMPLEX,color=(255,255,255),
                            thickness=2,fontScale=0.5)
            # image = cv2.line(image, (0, 600), (1500, 600), (0, 255, 0), 1)

    return image


def main():
    args = build_argparser()
    network = Inference()

    total_count = 0

    # load network
    network.load_network(args.m,device=args.d)

    # get network input shape
    input_shape = network.get_input_shape()

    if args.i == 'CAM':
        input_stream = 0
    else:
        input_stream = args.i

    # start read video
    cap = cv2.VideoCapture(input_stream)
    cap.open(args.i)
    while True:
        ret,frame = cap.read()
        if not ret:
            lg.error('Fail to load video/camera')
            break
        # process video frame
        p_frame = network.preprocess_input(frame,256,256)

        # start inference on each frame
        network.async_infer(p_frame)

        # check inference status is finished
        if network.wait() == 0:

            # get inference output
            output = network.get_ouput()

            # draw bboxes on frame
            frame = draw_bbox(frame,output,args.pt)

        # display frame
        cv2.imshow('frame',frame)
        
        if cv2.waitKey(25) & 0xff == ord('q'):
            break

if __name__ == '__main__':
    main()