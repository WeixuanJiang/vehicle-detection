from openvino.inference_engine import IECore,IENetwork
import os
import logging as lg
import sys
import cv2
import numpy as np

class Inference:
    def __init__(self):
        self.plugin = None
        self.status = None
        self.input_blob = None
        self.output_blob = None
        self.input_shape = None
        self.network = None
        self.exec_network = None
        self.inference_request = None

    def load_network(self,model,device='CPU',cpu_extenstion=None,num_requests=1):
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + '.bin'

        self.plugin = IECore()

        if device == 'CPU' and cpu_extenstion:
            self.plugin.add_extension(cpu_extenstion)
            lg.info('CPU extension loaded {}'.format(cpu_extenstion))

        try:
            self.network = IENetwork(model_xml,model_bin)
            lg.info('Model loaded to network')
        except:
            lg.error('Fail to load model to network, please check your model path')


        # check supported layers
        if 'CPU' in device:
            supported_layers = self.plugin.query_network(self.network,'CPU')
            unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]

            if len(unsupported_layers) > 0:
                lg.error('Found unsupported layers on device {} with {}'.format(device,unsupported_layers))

        lg.info('Support layers checked, passed')

        self.exec_network = self.plugin.load_network(self.network,device,num_requests=num_requests)
        lg.info('Network loaded')

        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))

        # get input shape
        self.input_shape = self.network.inputs[self.input_blob].shape

        return



    def preprocess_input(self,image,height,width):
        image = cv2.resize(image,(width,height))
        image = image.transpose((2,0,1))
        image = image.reshape((1,3,height,width))
        return np.array(image)

    def get_ouput(self):

        return self.exec_network.requests[0].outputs[self.output_blob]


    def wait(self):

        self.status = self.exec_network.requests[0].wait(-1)

        return self.status


    def async_infer(self,image,request_id=0):

        self.inference_request = self.exec_network.start_async(inputs={self.input_blob: image}
                                                               ,request_id=request_id)

        return self.inference_request


    def get_input_shape(self):
        return self.input_shape






















