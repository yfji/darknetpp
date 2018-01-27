export LD_LIBRARY_PATH=/home/zbox/Workspace/opencv-3.3.0/build/install/lib:/home/zbox/Workspace/darknetpp:$LD_LIBRARY_PATH
./darknet detect cfg/yolo.cfg weights/yolo.weights data/dog.jpg
