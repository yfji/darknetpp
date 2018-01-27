export LD_LIBRARY_PATH=/usr/local/lib:/home/zbox/Workspace/darknetpp:$LD_LIBRARY_PATH
./darknet detect cfg/yolo.cfg weights/yolo.weights data/dog.jpg
