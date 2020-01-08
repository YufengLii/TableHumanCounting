# -*- coding: utf-8 -*-

import sys
import time
import cv2
import multiprocessing as mp
from modules.dbutil import MySQLPlugin
from modules import mqutil
from my_utils import cvdraw, judge, myconfig

try:
    sys.path.append(myconfig.opPython)
    from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found.')
    raise e

camera_info_lists = myconfig.camera_info_lists
url_lists = myconfig.camera_info_lists

# print('Table calibration info: ')
# pprint(camera_info_lists)
# pprint(url_lists)


def image_put(q, Video_DIR, cameraID):

    cap = cv2.VideoCapture(Video_DIR)
    while True:
        q.put([cameraID, cap.read()[1]])
        q.get() if q.qsize() > 1 else time.sleep(0.01)


def inwhichTable(queue_list, Video_DIR_List, camera_ID_List):
    TableHumanKPTimeList = {
        '7': [],
        '10': [],
        '11': [],
        '12': [],
        '13': [],
    }

    TableHumanNumTimeList = {
        '7': [],
        '10': [],
        '11': [],
        '12': [],
        '13': [],
    }

    try:
        opWrapper = op.WrapperPython()
        opWrapper.configure(myconfig.op_params)
        opWrapper.start()
    except Exception as e:
        print("opWrapper init error!!!")
        print(e)
        sys.exit(-1)

    # mysql_q = mqutil.RSMQueue('cvstats')
    while True:
        for q in queue_list:
            # get image from q
            CurrCameraID, CurrFrame = q.get()
            print('------------------------------------------')
            print('\n\nRun On Camera IP: ' + '192.168.31.' + str(CurrCameraID))
            # human keypoints detection
            datum = op.Datum()
            datum.cvInputData = CurrFrame
            opWrapper.emplaceAndPop([datum])
            frame_keypoints = datum.poseKeypoints

            CurrCameraCali = camera_info_lists[str(CurrCameraID)]
            CurrCameraCaliInfo = CurrCameraCali['calibrations']

            # human in image

            for ETableCaliInfo in CurrCameraCaliInfo:

                if ETableCaliInfo['shape'] == 'ellipses':

                    if len(frame_keypoints.shape) == 0:
                        TableHumanNumTimeList = judge.updateHumanNumTimeList(
                            ETableCaliInfo['tableID'], 0,
                            TableHumanNumTimeList, myconfig.JudgeLength)
                        # TableHumanKPTimeList[str(CurrCameraID)] = []
                        break
                    CurrFrame = cvdraw.drawEllipseCalibtarion(
                        ETableCaliInfo, myconfig.TableAround, CurrFrame)

                    TableAround = myconfig.TableAround[str(
                        ETableCaliInfo['tableID'])]

                    print('\nRun On Table: ' + str(ETableCaliInfo['tableID']))

                    # List of [Neck, RHip, LHip, MidHip, RShoulder, LShoulder]
                    CurrTableHumanKP = []

                    for human_keypoints in frame_keypoints:

                        Neck = human_keypoints[1].tolist()
                        MidHip = human_keypoints[8].tolist()
                        RHip = human_keypoints[9].tolist()
                        LHip = human_keypoints[12].tolist()
                        RShoulder = human_keypoints[2].tolist()
                        LShoulder = human_keypoints[5].tolist()
                        RKnee = human_keypoints[10].tolist()
                        LKnee = human_keypoints[13].tolist()

                        usedKeyPointsList = [
                            Neck, RHip, LHip, MidHip, RShoulder, LShoulder,
                            RKnee, LKnee
                        ]

                        if judge.IsInEllipseTableArround(
                                ETableCaliInfo['tableID'], ETableCaliInfo,
                                usedKeyPointsList) is True:
                            CurrTableHumanKP.append(usedKeyPointsList)
                            print('In table ',
                                  str(ETableCaliInfo['tableID']),
                                  'Around',
                                  sep=' ')

                        print(str(len(CurrTableHumanKP)),
                              'Human around',
                              str(ETableCaliInfo['tableID']),
                              sep=' ')

                    if len(CurrTableHumanKP) == 0:
                        TableHumanNumTimeList = judge.updateHumanNumTimeList(
                            ETableCaliInfo['tableID'], 0,
                            TableHumanNumTimeList, myconfig.JudgeLength)
                        TableHumanKPTimeList[str(
                            ETableCaliInfo['tableID'])] = []
                        print('no human around Table',
                              str(ETableCaliInfo['tableID']),
                              'Table List Reset',
                              sep=' ')
                    else:
                        if len(TableHumanKPTimeList[str(
                                ETableCaliInfo['tableID'])]
                               ) == myconfig.KeypointListMaxLength:
                            print('桌子区域有人驻留，进入聚类分�?')
                            del TableHumanKPTimeList[str(
                                ETableCaliInfo['tableID'])][0]
                            TableHumanKPTimeList[str(
                                ETableCaliInfo['tableID'])].append(
                                    CurrTableHumanKP)
                            print('Table',
                                  str(ETableCaliInfo['tableID']),
                                  'List full Judge Ready!!!',
                                  sep=' ')
                            # pprint(TableHumanKPTimeList[str(ETableCaliInfo['tableID'])])
                            # 取出序列中的人体中心点MidHip
                            TrackingMainPostionNP, TrackingWholePostionNP = judge.HumanMainPostionExtract(
                                TableHumanKPTimeList[str(
                                    ETableCaliInfo['tableID'])])
                            # pprint(TrackingWholePostionNP)
                            # 聚类
                            lables, n_clusters_ = judge.DBscanCluster(
                                TrackingMainPostionNP, myconfig.ClusterEps,
                                myconfig.ClusterMinSample)
                            # 聚类结果数量
                            if lables.max() < 0:
                                print('Moving, not siting')
                                n_clusters_ = 0

                            if n_clusters_ > 0:
                                FinalHumanKpCoorList = judge.FinalPosition(
                                    lables, TrackingWholePostionNP)
                                # pprint(FinalHumanKpCoorList)
                                # CurrFrame = judge.FinalMainBodyRect(CurrFrame, FinalHumanKpCoorList)
                                # OverLapPercentageList, tableHumanKpCoorListOverlap = judge.OverLapPercentage(
                                #     ETableCaliInfo, FinalHumanKpCoorList,
                                #     myconfig.OverlapThreshold)

                                OverLapPercentageList, BetweenElipPercentageList, tableHumanKpCoorListOverlap = judge.OverLapPercentage_2(
                                    ETableCaliInfo, TableAround,
                                    FinalHumanKpCoorList,
                                    myconfig.OverlapThreshold,
                                    myconfig.BetweenElipseThreshold)
                                # print('OverLapPercentageList:',
                                #       OverLapPercentageList,
                                #       '%',
                                #       sep=' ')
                                tableHumanKpCoorListHWthresh = judge.HeightWidthThreshhold(
                                    tableHumanKpCoorListOverlap,
                                    ETableCaliInfo['long_axis'],
                                    ETableCaliInfo['long_axis'])
                                # CurrTableHumanNum = tableHumanKpCoorListHWthresh.shape[0]
                                FinalTableHumanList, FinalKneeCoorList = judge.KeenJudge(
                                    tableHumanKpCoorListHWthresh, TableAround,
                                    ETableCaliInfo)

                                TableHumanNumTimeList = judge.updateHumanNumTimeList(
                                    ETableCaliInfo['tableID'],
                                    FinalTableHumanList.shape[0],
                                    TableHumanNumTimeList,
                                    myconfig.JudgeLength)

                                if myconfig.INDEBUG is True:
                                    cvdraw.drawEllipseDebug(
                                        CurrFrame, lables,
                                        myconfig.OverlapThreshold,
                                        FinalHumanKpCoorList,
                                        OverLapPercentageList,
                                        BetweenElipPercentageList,
                                        FinalTableHumanList,
                                        TrackingMainPostionNP, ETableCaliInfo,
                                        FinalKneeCoorList)

                        else:
                            TableHumanNumTimeList = judge.updateHumanNumTimeList(
                                ETableCaliInfo['tableID'], 0,
                                TableHumanNumTimeList, myconfig.JudgeLength)
                            TableHumanKPTimeList[str(
                                ETableCaliInfo['tableID'])].append(
                                    CurrTableHumanKP)
                            print('Table',
                                  str(ETableCaliInfo['tableID']),
                                  'List not full Judge not Ready!!!',
                                  sep=' ')
            cv2.imshow(str(CurrCameraID), CurrFrame)
            cv2.waitKey(1)
            # judge.updateDataBase(mysql_q, TableHumanNumTimeList,
            #                      myconfig.JudgeLength)
            print('------------------------------------------')


def run_multi_camera():

    mp.set_start_method(method='spawn')  # init
    queues = [mp.Queue(maxsize=2) for _ in myconfig.Video_DIR_List]

    processes = [
        mp.Process(target=inwhichTable,
                   args=(queues, myconfig.Video_DIR_List,
                         myconfig.camera_ID_List))
    ]
    for queue, Video_DIR, cameraID in zip(queues, myconfig.Video_DIR_List,
                                          myconfig.camera_ID_List):
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
