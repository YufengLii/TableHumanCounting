# -*- coding: utf-8 -*-

import sys
import time
import cv2
import multiprocessing as mp
from modules.dbutil import MySQLPlugin
from modules import mqutil
from my_utils import cvdraw, judge, SquareConfig
import json

try:
    sys.path.append(SquareConfig.opPython)
    from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found.')
    raise e

with open("./data/squareCali.txt", 'r') as f:
    SquareCaliDict = json.loads(f.read())


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
    mysql_q = mqutil.RSMQueue('cvstats')
    TableHumanKPTimeList = SquareConfig.TableHumanKPTimeList
    TableHumanNumList = SquareConfig.TableHumanNumList

    for key, _ in TableHumanNumList.items():
        msg_str = {
            'areaID': key,
            'timestamp': int(time.time() * 1000),
            'num': 0,
        }
        mysql_q.publish(json.dumps(msg_str))        

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
            for STable in CurrCameraSquareCali:

                cvdraw.drawSquareTableInfo(CurrFrame, STable)

                TableID = str(STable['tableID'])
                PerTableHumanNum = TableHumanNumList[TableID][
                    'front'] + TableHumanNumList[TableID]['back']

                if len(frame_keypoints.shape) == 0:
                    TableHumanKPTimeList[TableID]['front'] = []
                    TableHumanKPTimeList[TableID]['back'] = []
                    continue

                CurrTableHumanKP = {'back': [], 'front': []}
                for human_keypoints in frame_keypoints:

                    usedKeyPointsList = [
                        human_keypoints[2].tolist(),
                        human_keypoints[5].tolist()
                    ]

                    if judge.IsInSquareTF(STable, usedKeyPointsList) is True:
                        CurrTableHumanKP['front'].append(usedKeyPointsList)

                    if judge.IsInSquareTB(STable, usedKeyPointsList) is True:
                        CurrTableHumanKP['back'].append(usedKeyPointsList)

                TableHumanKPTimeList, FrontReady, BackReady = judge.UpdateTableHumanKPTimeList(
                    TableHumanKPTimeList, SquareConfig.KeypointListMaxLength,
                    CurrTableHumanKP, STable)

                if FrontReady is True:
                    FrontHumanKPTimeList = TableHumanKPTimeList[TableID][
                        'front']
                    FrontTrackingKp, FrontKpList = judge.SquareTrackingKpExtract(
                        FrontHumanKPTimeList)
                    Flabels, Fn_clusters_ = judge.DBscanCluster(
                        FrontTrackingKp, SquareConfig.ClusterEps,
                        SquareConfig.ClusterMinSample)

                    TableHumanNumList[TableID]['front'] = max(Flabels) + 1
                    del TableHumanKPTimeList[TableID]['front'][0]
                    if SquareConfig.INDEBUG is True:

                        CurrFrame = cvdraw.drawSquareTableDebug(
                            CurrFrame, FrontTrackingKp, FrontKpList, Flabels,
                            STable, 'Front')

                if BackReady is True:
                    BackHumanKPTimeList = TableHumanKPTimeList[TableID]['back']
                    BackTrackingKp, BackKpList = judge.SquareTrackingKpExtract(
                        BackHumanKPTimeList)
                    Blabels, Bn_clusters_ = judge.DBscanCluster(
                        BackTrackingKp, SquareConfig.ClusterEps,
                        SquareConfig.ClusterMinSample)

                    TableHumanNumList[TableID]['back'] = max(Blabels) + 1
                    del TableHumanKPTimeList[TableID]['back'][0]
                    if SquareConfig.INDEBUG is True:
                        CurrFrame = cvdraw.drawSquareTableDebug(
                            CurrFrame, BackTrackingKp, BackKpList, Blabels, STable,
                            "Back")
                CurrTableHumanNum = TableHumanNumList[TableID][
                    'back'] + TableHumanNumList[TableID]['front']
                if CurrTableHumanNum != PerTableHumanNum:
                    print("DB Update")
                    msg_str = {
                        'areaID': TableID,
                        'timestamp': int(time.time() * 1000),
                        'num': max(Blabels)+1,
                    }
                    mysql_q.publish(json.dumps(msg_str))
            if SquareConfig.INDEBUG is True:

                cv2.imshow(str(CurrCameraID), CurrFrame)
                cv2.waitKey(1)

            print('\n\n------------------------------------------')


def run_multi_camera():

    mp.set_start_method(method='spawn')  # init
    queues = [mp.Queue(maxsize=5) for _ in SquareConfig.Video_DIR_List]

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
