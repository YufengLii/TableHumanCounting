from my_utils import cvdraw

op_params = dict()
op_params["model_folder"] = "/home/feng/Documents/openpose/models/"
op_params["model_pose"] = "BODY_25"
op_params["net_resolution"] = "-1x368"
op_params["num_gpu"] = 1
op_params["num_gpu_start"] = 0

opPython = '/home/feng/Documents/openpose/build/python'


INDEBUG = True
camera_ID_List = [121]
Video_DIR_List = [
    "./data/121.mp4",
]
OverlapThreshold = 40
BetweenElipseThreshold = 35
KeypointListMaxLength = 10
working_cameraID = [109, 121, 110, 112, 118]
ClusterEps = 70
ClusterMinSample = 10
JudgeLength = 3
TableAround = {
    '7': [20, 0, 2.5, 3.8],
    '10': [0, 100, 2.2, 3.5],
    '11': [0, 100, 2.2, 3.5],
    '12': [25, 120, 2.2, 2.2],
    '13': [60, 135, 2.1, 3]
}

camera_info_lists, url_lists = cvdraw.get_cali_data_from_file(working_cameraID)
