from my_utils import cvdraw

op_params = dict()
op_params["model_folder"] = "/home/feng/Documents/openpose/models/"
op_params["model_pose"] = "BODY_25"
op_params["net_resolution"] = "-1x368"
op_params["num_gpu"] = 1
op_params["num_gpu_start"] = 0



INDEBUG= True
camera_ID_List= [121, 118]
Video_DIR_List = [
    "./data/121.mp4",
    "./data/118.avi",
]
OverlapThreshold = 40
KeypointListMaxLength = 10
working_cameraID = [109, 121, 110, 112, 118]

camera_info_lists, url_lists = cvdraw.get_cali_data_from_file(working_cameraID)
