!#/
python notebooks/huduki.py
./darknet detector test cfg/drone.data cfg/yolo-drone.cfg weights/yolo-drone.weights -dont_show -ext_output < darknet/data/train.txt > result.txt

