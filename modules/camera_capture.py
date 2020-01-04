#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'werewolfLu'


import cv2
import time


def camera_read(camera, queue, frame_rate=None):
    skip_frame = 1
    if frame_rate is not None:
        skip_frame = int(camera['frame_rate'] / frame_rate)
        print('skip frame', skip_frame)
        if skip_frame <= 0:
            print("[Camera] Argument error: invalid frame_rate", frame_rate, ", will use ", camera["frame_rate"])

    ind = 0
    while True:
        try:
            c = cv2.VideoCapture(camera['url'])

            if c.isOpened():

                print("cameraID " + str(camera['name']) + ': Open success')
                print("FPS ", c.get(cv2.CAP_PROP_FPS), c.get(cv2.CAP_PROP_POS_MSEC))

                if camera.get('resolution'):
                    w,h = camera['resolution'].split('x')
                    c.set(cv2.CAP_PROP_FRAME_WIDTH, int(w))
                    c.set(cv2.CAP_PROP_FRAME_HEIGHT, int(h))

                count = 0
                failcounts = 0
                t1 = time.time()

                while True:

                    success, frame = c.read()
                    if success:
                        count += 1
                        failcounts = 0
                        if count % skip_frame != 0:
                            continue

                        ind += 1
                        try:
                            queue.put([camera['name'], ind, frame])
                            while queue.qsize() > 1:
                                queue.get(block=False)
                            # queue.get(block=False) if queue.qsize() > 1 else time.sleep(0.001)
                            # print(time.time() - t1)
                            # t1 = time.time()
                        except Exception as e:
                            pass
                    else:

                        if not c.isOpened():
                            print("camera", camera['name'], "closed, retry in 10 sec")
                            time.sleep(10)
                            c.release()
                            break
                        else:
                            print("camera", camera['name'], "return null")
                            time.sleep(.01)
                            failcounts += 1
                            if failcounts > 10:
                                c.release()
                                break
            else:
                print("camera", camera['name'], "is not opened, retry in 30 sec")
                time.sleep(30)
                c.release()

        except Exception as e:
            print("Camera Exception", e)
            try:
                c.release()
            except Exception as e:
                pass
            time.sleep(1)


def img_show(window, q):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    while True:
        try:
            channel, ind, frame = q.get()
            print(1111, type(frame))
            # cv2.imshow(window, frame)
            result, encimg = cv2.imencode('.jpg', frame, encode_param)
            open('/Users/ping/Downloads/b.jpeg', 'wb').write(encimg)

        except Exception as e:
            print('IMG Exception', e)
            pass


def forwared(q):
    from modules.mqutil import RSMQueue
    from modules.baidutil import img_encodeb64
    t = RSMQueue('realSense', host='192.168.31.184')
    while True:
        try:
            channel, ind, frame = q.get()
            # print(1111, type(frame))
            # # cv2.imshow(window, frame)
            # result, encimg = cv2.imencode('.jpg', frame, encode_param)
            # open('/Users/ping/Downloads/b.jpeg', 'wb').write(encimg)
            t.publish(img_encodeb64(frame))

        except Exception as e:
            print('IMG Exception', e)
            pass


if __name__ == '__main__':

    import multiprocessing as mp

    mp.set_start_method('spawn')

    manager = mp.Manager()
    q = manager.Queue()

    cameras = {
        'native': {
            'name': 'native',
            'url': 0,
            'frame_rate': 30,
            'resolution': '1280x720'
        },
        'channel1': {
            'name': 'channel1',
            'url': 'rtsp://guest:password_1234@192.168.31.222/Streaming/Channels/101',
            'frame_rate': 25,
            # 'resolution': '1280x720'
        },
        'channel2': {
            'name': 'channel2',
            'url': 'rtsp://guest:password_1234@192.168.31.101/cam/realmonitor?channel=1&subtype=0',
            'frame_rate': 25,
            # 'resolution': '1280x720'
        },
        'hik_1': {
            'name': 'hik_1',
            'url': 'rtsp://guest:password_1234@192.168.31.50/Streaming/Channels/101',
            'frame_rate': 25,
            # 'resolution': '1280x720'
        },
        'hik_2': {
            'name': 'hik_2',
            'url': 'rtsp://guest:password_1234@192.168.31.50/Streaming/Channels/201',
            'frame_rate': 25,
            # 'resolution': '1280x720'
        },
        'hik_3': {
            'name': 'hik_3',
            'url': 'rtsp://guest:password_1234@192.168.31.50/Streaming/Channels/301',
            'frame_rate': 25,
            # 'resolution': '1280x720'
        },
        'hik_4': {
            'name': 'hik_4',
            'url': 'rtsp://guest:password_1234@192.168.31.50/Streaming/Channels/401',
            'frame_rate': 25,
            # 'resolution': '1280x720'
        },

    }

    # processes = [mp.Process(target=camera_read, args=(cameras['channel1'], q, 25)),
    #              mp.Process(target=img_show, args=('camera', q))]

    processes = [mp.Process(target=camera_read, args=(cameras['channel1'], q)),
                 mp.Process(target=forwared, args=(q,))]


    [p.start() for p in processes]
    [p.join() for p in processes]