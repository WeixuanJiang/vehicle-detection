##Vehicle Detection - OpenVINO

## Project Set Up and Installation
You will need to download the OpenVINO toolkit in order to run this model
please visit: https://docs.openvinotoolkit.org/latest/index.html 

Setup environment and dependencies:

Source to the current directory and run

    pip install -r requirements.txt
    cd <your path>\Intel\openvino_2021.1.110\bin
    setupvars.bat
    
Then you will need run a demo to test whether the environment is set up properly

    cd <your path>\Intel\openvino_2021.1.110\deployment_tools\demo\
    demo_squeezenet_download_convert_run.bat -d CPU 

Download pretrained models

    cd <your path>\Intel\openvino_2021.1.110\deployment_tools\tools\model_downloader
    python downloader.py --name vehicle-detection-0200 --o ./model/
    
## Demo
For running the model please follow the next command

    cd <your path>\Intel\openvino_2021.1.110\bin
    setupvars.bat
    python --main.py --m ./model/vehicle-detection-0200/FP16/vehicle-detection-0200.xml --i ./Demo/demo.mp4

## Documentation
Command line Arguments

    python --main.py
    
    --m path for face detection model
    
    --d  Device for model to run inference, default is CPU
    
    --i path for input data, default is to use camera
    
    --pt probability threhold for model to run inferencing, default is 0.5