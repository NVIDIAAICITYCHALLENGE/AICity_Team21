# NVIDIA AI City Challenge. Team 21.

### Synopsis

Track 1 utilized the Darknet framework with Yolo object detection. We achived 2nd place in mean average precision for the AI city challenge using this network and training parameters. 

You will need to build darknet in order to train and run inference on the models. 

### Motivation
we used Yolo since it is known for its impressive speed and accuracy. It is perfect to transfer to an edge device such as the TX2 due to the reasonable FPS on live video. 

### Installation

Track 1 instructions.

0. clone repo
1. cd track1
2. make
3. download model weights from this link: https://drive.google.com/drive/folders/0B_Wc2TdfdVh5UmlKNl9mX2FKOHc?usp=sharing

### Code Example

* run inference on a single image. output predicted bounding boxes overlaid on top of source image. output predictions.jpg will be current directory.
 `./darknet detect test data/aic540.data cfg/aic540.cfg aic540_final.weights /path/to/image.jpg `

* run inference on a batch of images, with corresponding text files
  * `cd track1`
  * `mkdir results`
  * create a list of images we want to detect `find /datasets/aic480/test/images -type f -name "*.jpeg > aic480_test_set.txt"`
  * run detection with 10% minimum confidence threshold `./darknet detector test_file data/aic480.data cfg/aic480.cfg weights/yolo-object_30000.weights test_sets/aic480_test_set.txt -thresh 0.10 -outdir results`

* train darknet from pretrained convolutional weights
`./darknet detector train data/aic1080.data data/aic1080.cfg /data/group1/weights/darknet19_448.conv.23`

* run inference on video and write video to disk.
`./darknet detector demo data/aic540.data data/aic540.cfg weights/aic540_final.weights /home/charles/Videos/vlc-record-2017-08-01-23h50m10s-walsh_santomas_20170602_007.mp4 `
  * output will be saved as Output.avi, 


### Contributors

 * [Charles MacKay](https://github.com/ctmackay)
 * [Niveditha Bhandary](https://github.com/nivedithabhandary)
 * Alex Richards
 * Ji Tong
 * [David Anastasiu (Advisor)](https://github.com/davidanastasiu)


### License
