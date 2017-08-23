## NVIDIA AI City Challenge. Team 21.

### Synopsis

Track 1 utilized the Darknet framework with Yolo object detection. We achived 2nd place in mean average precision for the AI city challenge using this network and training parameters. 

You will need to build darknet in order to train and run inference on the models. 

## Motivation
we used Yolo since it is known for its impressive speed and accuracy. It is perfect to transfer to an edge device such as the TX2 due to the reasonable FPS on live video. 

## Installation

Track 1 instructions.

0. clone repo
1. cd track1
2. make
3. download model weights. 

## Code Example
4. run inference on the detected model. output predicted bounding boxes overlaid on top of source image:
 ./darknet detect test data/aic540.data cfg/aic540.cfg aic540_final.weights /path/to/image.jpg 
 

## Contributors

 [Charles MacKay](https://github.com/ctmackay)
 

## License
