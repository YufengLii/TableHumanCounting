# -*- coding: utf-8 -*-

import sys
import time
import cv2
import multiprocessing as mp
from modules.dbutil import MySQLPlugin
from modules import mqutil
from my_utils import cvdraw, judge, SquareConfig
from pprint import pprint
import json

try:
    sys.path.append(SquareConfig.opPython)
    from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found.')
    raise e

with open("./data/squareCali.txt", 'r') as f:
    SquareCaliDict = json.loads(f.read())

camera_info_lists = SquareConfig.camera_info_lists
url_lists = SquareConfig.url_lists

# print('Table calibration info: ')
# pprint(camera_info_lists)
# pprint(url_lists)


def image_put(q, Video_DIR, cameraID):

    cap = cv2.VideoCapture(Video_DIR)
    while True:
        q.put([cameraID, cap.read()[1]])
        q.get() if q.qsize() > 1 else time.sleep(0.01)


def inwhichTable(queue_list, Video_DIR_List, camera_ID_List):

    try:
        opWrapper = op.WrapperPython()
        opWrapper.configure(SquareConfig.op_params)
        opWrapper.start()
    except Exception as e:
        print("opWrapper init error!!!")
        print(e)
        sys.exit(-1)

    TableHumanKPTimeList = SquareConfig.TableHumanKPTimeList

    while True:

        for q in queue_list:
            CurrCameraID, CurrFrame = q.get()
            CurrFrame = cvdraw.image_preprocess(
                CurrFrame, SquareConfig.cut[str(CurrCameraID)])
            CurrCameraSquareCali = SquareCaliDict[str(CurrCameraID)]
            print('------------------------------------------')
            print('\n\nRun On Camera IP: ' + '192.168.31.' + str(CurrCameraID))

            # human keypoints detection
            datum = op.Datum()
            datum.cvInputData = CurrFrame
            opWrapper.emplaceAndPop([datum])
            frame_keypoints = datum.poseKeypoints

            # human in image
            for STableCaliInfo in CurrCameraSquareCali:
                cvdraw.drawSquareTableInfo(CurrFrame, STableCaliInfo)
                if len(frame_keypoints.shape) == 0:
                    print("no human detected")
                    TableHumanKPTimeList[str(
                        STableCaliInfo['tableID'])]['front'] = []
                    TableHumanKPTimeList[str(
                        STableCaliInfo['tableID'])]['back'] = []
                    continue
                CurrTableHumanKP = {'back': [], 'front': []}

                for human_keypoints in frame_keypoints:

                    Neck = human_keypoints[1].tolist()
                    MidHip = human_keypoints[8].tolist()
                    RHip = human_keypoints[9].tolist()
                    LHip = human_keypoints[12].tolist()
                    RShoulder = human_keypoints[2].tolist()
                    LShoulder = human_keypoints[5].tolist()

                    usedKeyPointsList = [
                        Neck, RHip, LHip, MidHip, RShoulder, LShoulder
                    ]

                    if judge.IsInSquareTableFront(STableCaliInfo,
                                                  usedKeyPointsList) is True:
                        CurrTableHumanKP['front'].append(usedKeyPointsList)
                        cv2.circle(CurrFrame,
                                   (int(LShoulder[0]), int(LShoulder[1])), 5,
                                   (0, 0, 255), -1)
                        cv2.circle(CurrFrame,
                                   (int(RShoulder[0]), int(RShoulder[1])), 5,
                                   (0, 0, 255), -1)

                    if judge.IsInSquareTableBack(STableCaliInfo,
                                                 usedKeyPointsList) is True:
                        CurrTableHumanKP['back'].append(usedKeyPointsList)
                        cv2.circle(CurrFrame,
                                   (int(LShoulder[0]), int(LShoulder[1])), 5,
                                   (0, 0, 255), -1)
                        cv2.circle(CurrFrame,
                                   (int(RShoulder[0]), int(RShoulder[1])), 5,
                                   (0, 0, 255), -1)
                TableHumanKPTimeList, FrontReadyAnalyze, BackReadyAnalyze = judge.UpdateTableHumanKPTimeList(
                    TableHumanKPTimeList, SquareConfig.KeypointListMaxLength,
                    CurrTableHumanKP, STableCaliInfo)

                if FrontReadyAnalyze is True:
                    FrontHumanKPTimeList = TableHumanKPTimeList[str(
                        STableCaliInfo['tableID'])]['front']
                    FrontTrackingKp, FrontTrackingWholekp = judge.SquareTrackingKpExtract(
                        FrontHumanKPTimeList)
                    Flabels, Fn_clusters_ = judge.DBscanCluster(
                        FrontTrackingKp, SquareConfig.ClusterEps,
                        SquareConfig.ClusterMinSample)
                    CurrFrame = cvdraw.drawSquareTableDebug(
                        CurrFrame, FrontTrackingKp, Flabels)

                    if Fn_clusters_ > 1:
                        print("Table " + str(STableCaliInfo['tableID']) +
                              " Front human num is�� " + str(Fn_clusters_ - 1))
                    else:
                        print("Moving")
                    del TableHumanKPTimeList[str(
                        STableCaliInfo['tableID'])]['front'][0]

                if BackReadyAnalyze is True:
                    BackHumanKPTimeList = TableHumanKPTimeList[str(
                        STableCaliInfo['tableID'])]['back']
                    BackTrackingKp, BackTrackingWholekp = judge.SquareTrackingKpExtract(
                        BackHumanKPTimeList)
                    Blabels, Bn_clusters_ = judge.DBscanCluster(
                        BackTrackingKp, SquareConfig.ClusterEps,
                        SquareConfig.ClusterMinSample)

                    CurrFrame = cvdraw.drawSquareTableDebug(
                        CurrFrame, BackTrackingKp, Blabels)

                    if Bn_clusters_ > 1:
                        print("Table " + str(STableCaliInfo['tableID']) +
                              " Back human num is�� " + str(Bn_clusters_ - 1))
                    else:
                        print("Moving")
                    del TableHumanKPTimeList[str(
                        STableCaliInfo['tableID'])]['back'][0]

            cv2.imshow(str(CurrCameraID), CurrFrame)
            cv2.waitKey(1)

            print('------------------------------------------')


def run_multi_camera():

    mp.set_start_method(method='spawn')  # init
    queues = [mp.Queue(maxsize=3) for _ in SquareConfig.Video_DIR_List]

    processes = [
        mp.Process(target=inwhichTable,
                   args=(queues, SquareConfig.Video_DIR_List,
                         SquareConfig.camera_ID_List))
    ]
    for queue, Video_DIR, cameraID in zip(queues, SquareConfig.Video_DIR_List,
                                          SquareConfig.camera_ID_List):
        processes.append(
            mp.Process(target=image_put, args=(queue, Video_DIR, cameraID)))

    for process in processes:
        process.daemon = True  # setattr(process, 'deamon', True)
        process.start()
    for process in processes:
        process.join()


if __name__ == '__main__':
    # run_single_camera()
    run_multi_camera()
    pass
