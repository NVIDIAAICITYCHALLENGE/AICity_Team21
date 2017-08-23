NVIDIA AI City Challenge. Team 21.

Track 1 utilized the Darknet framework with Yolo object detection. You will need to build darknet in order to train and run inference on the models. 

Track 1 instructions.
0. clone repo
1. cd track1
2. make
3. download model weights. 
4. run inference on the detected model. output predicted bounding boxes overlaid on top of source image:
 ./darknet detect test data/aic540.data cfg/aic540.cfg aic540_final.weights /path/to/image.jpg 
