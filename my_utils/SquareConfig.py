from my_utils import cvdraw

op_params = dict()
op_params["model_folder"] = "/home/cv/Documents/lf/1225/openpose/models/"
op_params["model_pose"] = "BODY_25"
op_params["net_resolution"] = "-1x368"
op_params["num_gpu"] = 1
op_params["num_gpu_start"] = 1

opPython = '/home/cv/Documents/lf/1225/openpose/build/python'

INDEBUG = False
# camera_ID_List = [105, 106, 107, 108, 110, 112, 114, 119]
# Video_DIR_List = [
#     "./data/105.mp4",
#     "./data/106.mp4",
#     "./data/107.mp4",
#     "./data/108.mp4",
#     "./data/110.mp4",
#     "./data/112.mp4",
#     "./data/114.mp4",
#     "./data/119.mp4",
# ]
working_cameraID = [105, 106, 107, 108, 112, 114, 119]

camera_info_lists, url_lists = cvdraw.get_cali_data_from_file(working_cameraID)
camera_ID_List = [105, 106, 107, 108, 112, 114, 119]
Video_DIR_List = [
    url_lists['105'],
    url_lists['106'],
    url_lists['107'],
    url_lists['108'],
    url_lists['112'],
    url_lists['114'],
    url_lists['119'],
]

KeypointListMaxLength = 10
ClusterEps = 150
ClusterMinSample = 8


cut = {
    '105': [1000, 2440, 0, 1440, 90],
    '106': [0, 2560, 0, 1440, -90],
    '107': [900, 2340, 0, 1440, 90],
    '108': [80, 1520, 0, 1440, -90],
    '110': [0, 2560, 0, 1440, 0],
    '112': [0, 2560, 0, 1440, 0],
    '114': [800, 2240, 0, 1440, 0],
    '119': [400, 1840, 0, 1440, 0],
}
TableHumanKPTimeList = {
    '1': {
        'front': [],
        'back': []
    },
    '2': {
        'front': [],
        'back': []
    },
    '3': {
        'front': [],
        'back': []
    },
    '4': {
        'front': [],
        'back': []
    },
    '5': {
        'front': [],
        'back': []
    },
    '6': {
        'front': [],
        'back': []
    },
    '8': {
        'front': [],
        'back': []
    },
    '9': {
        'front': [],
        'back': []
    },
    '20': {
        'front': [],
        'back': []
    },
    '21': {
        'front': [],
        'back': []
    },
    '23': {
        'front': [],
        'back': []
    },
    '24': {
        'front': [],
        'back': []
    },
}



TableHumanNumList = {
    '1': {
        'front': 0,
        'back': 0
    },
    '2': {
        'front': 0,
        'back': 0
    },
    '3': {
        'front': 0,
        'back': 0
    },
    '4': {
        'front': 0,
        'back': 0
    },
    '5': {
        'front': 0,
        'back': 0
    },
    '6': {
        'front': 0,
        'back': 0
    },
    '8': {
        'front': 0,
        'back': 0
    },
    '9': {
        'front': 0,
        'back': 0
    },
    '20': {
        'front': 0,
        'back': 0
    },
    '21': {
        'front': 0,
        'back': 0
    },
    '23': {
        'front': 0,
        'back': 0
    },
    '24': {
        'front': 0,
        'back': 0
    },
}
