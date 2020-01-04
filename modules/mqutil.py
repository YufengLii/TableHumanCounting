#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'werewolfLu'


import time
from threading import Thread
from rsmq import RedisSMQ
from rsmq.consumer import RedisSMQConsumer, RedisSMQConsumerThread
try:
    from modules.mlogger import mlogger as logging
except ImportError as e:
    import logging


DEFAULT = {
    'server': '192.168.31.33'
}



class RSMQueue(object):
    _msg = []

    def __init__(self, qname, host=DEFAULT['server']):
        self.host = host
        self.qname = qname
        self.queue = RedisSMQ(host=host, qname=qname)
        self.consumer = None
        self.callback = None

        try:
            self.queue.deleteQueue().execute()
        except Exception as e:
            logging.error('[Exception] RSMQueue deleteQueue: %s', e)
            print('[Exception] RSMQueue deleteQueue: %s', e)

        try:
            self.queue.createQueue(delay=0).maxsize(-1).vt(0).execute()
        except Exception as e:
            logging.error('[Exception] RSMQueue createQueue: %s', e)
            print('[Exception] RSMQueue createQueue: %s', e)

    def set_callback(self, callback):
        self.callback = callback

    def publish(self, message):
        message_id = self.queue.sendMessage(delay=0).message(message).execute()
        self._msg.append(message_id)
        while len(self._msg) > 10:
            print(self._msg)
            try:
                self.queue.deleteMessage(id=self._msg[0]).execute()
                del self._msg[0]
            except Exception as e:
                logging.error('[Exception] RSMQueue publish: %s', e)
                print('[Exception] RSMQueue publish: %s', e)

        return message_id

    def deleteMessage(self, mid):
        return self.queue.deleteMessage(id=mid).execute()

    def subscribe1(self, qname, callback):
        self.consumer = RedisSMQConsumerThread(qname, callback, host=DEFAULT['server'])
        self.consumer.start()
        return self.consumer

    def receiveMessage(self, callback):
        try:
            id, message, rc, ts = self.queue.popMessage().execute()
            if callback and callable(callback):
                callback(message)
        except Exception as e:
            print('[Exception] receivemessage', e)

    def subscribe(self, callback, freq=10):
        queue = self.queue

        def f(callback):
            while True:
                try:
                    rt = queue.popMessage().execute()
                    print(rt)
                    if rt['id'] and callback and callable(callback):
                        callback(rt['message'])
                except Exception as e:
                    print('[Exception] receivemessage', e)
                    pass
                time.sleep(1/freq)

        t = Thread(target=f, args=(callback,))
        t.start()
        return t

    def cancel_subscribe(self):
        if self.consumer:
            self.consumer.stop()

    def peak(self):
        def _peak(id, message, rc, ts):
            print("\t\tpeak", id, message, rc, ts)
            time.sleep(0.1)
            return False
        self.subscribe( _peak)


def print_out(message):
    print("receive", message)
    return True


if __name__ == '__main__':

    q1 = RSMQueue('test1')
    q2 = RSMQueue('test2')



    # q.peak('test')
    # t1 = q1.subscribe(print_out)
    #
    # t2 = q2.subscribe(print_out)

    # t1.join()
    # t2.join()


    while True:
        msg = str(time.time())

        mid = q1.publish(msg)
        print('publish 1', mid, msg)

        mid = q2.publish(msg)
        print('publish 2', mid, msg)

        time.sleep(.1)
    #
    # t.join()


    # # example
    # FACE_IMG_QUEUE = "realSense"
    #
    # queue = RSMQueue(host="192.168.31.184", qname=FACE_IMG_QUEUE)
    # queue.publish('img')
    #
    #
    # FACE_IMG_QUEUE = "faceDetector"
    #
    # queue2 = RSMQueue(host="192.168.31.184", qname=FACE_IMG_QUEUE)
    # queue2.publish([{"location": [176, 814, 412, 578], "depth": 0.14561183750629425, "user_name": "\u6731\u742a\u742a", "face_score": 80, "gender": "male", "age": 20, "glasses": "common", "emotion": "neutral"}, {"location": [600, 202, 796, 6], "depth": 0.2120695263147354, "user_name": "\u6731\u742a\u742a", "face_score": 80, "gender": "male", "age": 20, "glasses": "common", "emotion": "neutral"}, {"location": [390, 446, 528, 308], "depth": 0.2737625539302826, "user_name": "\u6731\u742a\u742a", "face_score": 80, "gender": "male", "age": 20, "glasses": "common", "emotion": "neutral"}])
    #


