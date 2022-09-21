import cv2 
import zmq 

import json 
import pickle 

import operator as op 
import itertools as it, functools as ft 

import numpy as np 
import multiprocessing as mp 

from schema import ZMQModel 
from libraries.log import logger 
from typing import List, Tuple, Dict, Any  

def display_message(bgr_image:np.ndarray, message:str, color:Tuple[int, int, int], font_scale:int=1, font_thickness:int=1, font_face=cv2.FONT_HERSHEY_SIMPLEX):
    h, w, _ = bgr_image.shape 
    cx, cy = w // 2, h // 2
    (tw, th), tb = cv2.getTextSize(message, font_face, font_scale, font_thickness)
    cv2.putText(
        img=bgr_image, 
        text=message, 
        org=(cx - tw // 2, cy + th // 2 + tb), 
        fontFace=font_face, 
        fontScale=font_scale, 
        thickness=font_thickness, 
        color=color
    )

def distance(box0:Tuple[int, int, int, int], box1:Tuple[int, int, int, int]) -> float:
    pnt0 = np.array([box0[0] + box0[2] // 2, box0[1] + box0[3] // 2])
    pnt1 = np.array([box1[0] + box1[2] // 2, box1[1] + box1[3] // 2])
    return np.sqrt(np.sum((pnt1 - pnt0) ** 2) + 1e-8)

def update_trackers(trackers:List[Tuple[str, Any]], bgr_image:np.ndarray, condition:mp.Condition, flag:mp.Value, message_queue:mp.Queue) -> List[Tuple[str, Any]]:
    accumulator = []
    for region_id, tracker in trackers:
        tracking_status, predicted_roi = tracker.update(bgr_image)
        if tracking_status:
            predicted_roi = list(map(int, predicted_roi))
            message_queue.put({
                'type': ZMQModel.UPDATE_ROI,
                'data': {
                    'region_id': region_id, 
                    'coordinates': predicted_roi
                }
            }) 
            accumulator.append((region_id, tracker))
        else:
            message_queue.put({
                'type': ZMQModel.STOP_TRACKING,
                'data': {
                    'region_id': region_id, 
                    'coordinates': None
                }
            }) 
    
    condition.acquire()
    with flag.get_lock():
        flag.value = flag.value + 1 
    condition.notify_all()  # notify the server to check if the condition is still valid 
    condition.release()

    return accumulator

def subscriber_handshake(subscriber_socket:zmq.Socket, timeout:int) -> int:
    subscriber_poller_status = subscriber_socket.poll(timeout)
    if subscriber_poller_status == zmq.POLLIN:
        topic, _ = subscriber_socket.recv_multipart()
        if topic == ZMQModel.HANDSHAKE:
            return 1 
    return 0 

def dealer_handshake(dealer_socket:zmq.Socket, timeout:int) -> int:
    dealer_socket.send_multipart([b''], flags=zmq.SNDMORE)
    dealer_socket.send_pyobj({'type': ZMQModel.HANDSHAKE, 'data': ''})
    dealer_poller_status = dealer_socket.poll(timeout)
    if dealer_poller_status == zmq.POLLIN: 
        _, response_from_router = dealer_socket.recv_multipart()
        if response_from_router == ZMQModel.ACCEPTED:
            return 1 
    return 0 

def in_poller_map(socket:zmq.Socket, map_socket2events:Dict[zmq.Socket, int]) -> int:
    retrieved_status = map_socket2events.get(socket, None)
    if retrieved_status is not None: 
        if retrieved_status == zmq.POLLIN:
            return 1 
    return 0 

def worker(worker_id:int, router_address:str, publisher_address:str, timeout:int, tracker_type:str, readyness:mp.Event, condition:mp.Condition, flag:mp.Value, message_queue:mp.Queue):
    ZEROMQ_INIT = 0
    try:
        ctx = zmq.Context()
        dealer_socket:zmq.Socket = ctx.socket(zmq.DEALER)
        subscriber_socket:zmq.Socket = ctx.socket(zmq.SUB)

        dealer_socket.setsockopt_string(zmq.IDENTITY, f'{worker_id:03d}')
        dealer_socket.connect(router_address)

        subscriber_socket.connect(publisher_address)
        subscriber_socket.setsockopt_string(zmq.SUBSCRIBE, '')  # subscribe to all topics 

        readyness.wait(timeout=5)  # wait the signal from server
        if not readyness.is_set():
            raise Exception(f'worker {worker_id:03d} something wrong happen to the server (take too long)')

        subscriber_handshake_status = subscriber_handshake(subscriber_socket, timeout)  # wait 5s 
        if subscriber_handshake_status == 0:
            raise Exception(f'worker {worker_id:03d} subscriber_socket was not able to establish connection to {publisher_address}')
        logger.debug(f'worker {worker_id:03d} subscriber_socket has established connection to {publisher_address}')

        dealer_handshake_status = dealer_handshake(dealer_socket, timeout)  # wait 5s 
        if dealer_handshake_status == 0:
            raise Exception(f'worker {worker_id:03d} dealer_socket was not able to establish connection to {router_address}')
        logger.debug(f'worker {worker_id:03d} dealer_socket has established connection to {router_address}')

        poller = zmq.Poller()
        poller.register(dealer_socket, zmq.POLLIN)
        poller.register(subscriber_socket, zmq.POLLIN)
        ZEROMQ_INIT = 1  # zeromq ressources were initialized 

        trackers = []
        bgr_image = None 
        fn_template = f'Tracker{tracker_type}_create'

        keep_tracking = True 
        while keep_tracking:
            map_socket2events = dict(poller.poll(100))
            dealer_poller_status = in_poller_map(dealer_socket, map_socket2events)
            if dealer_poller_status == 1: 
                _, message_from_router = dealer_socket.recv_multipart()
                message = pickle.loads(message_from_router)
                if message['type'] == ZMQModel.TRACKING_REQ:
                    logger.debug(f'worker {worker_id:03d} has got a request for tracking')
                    bgr_image = message['data']['bgr_image']
                    tracker = op.attrgetter(fn_template)(cv2)()  # build tracker 
                    tracker.init(bgr_image, message['data']['coordinates'])
                    trackers.append((message['data']['region_id'], tracker))
                    
                    dealer_socket.send_multipart([b''], flags=zmq.SNDMORE)
                    dealer_socket.send_pyobj({  # send confirmation to router | server will increment the weights of this worker 
                        'type': ZMQModel.TRACKING_ACK, 
                        'data': ZMQModel.ACCEPTED
                    })

            subscriber_poller_status = in_poller_map(subscriber_socket, map_socket2events)
            if subscriber_poller_status == 1: 
                topic, message_from_publisher = subscriber_socket.recv_multipart()
                if topic == ZMQModel.STREAM:
                    bgr_image = pickle.loads(message_from_publisher)
                    if bgr_image is not None:
                        trackers = update_trackers(trackers, bgr_image, condition, flag, message_queue)
                if topic == ZMQModel.QUIT:
                    keep_tracking = False 
                    logger.debug(f'worker {worker_id:03d} has received the quit signal')
        # end loop multi object tracking ...

    except KeyboardInterrupt:
        pass 
    except Exception as e:
        logger.error(e) 
    finally:
        if ZEROMQ_INIT == 1:
            poller.unregister(subscriber_socket)
            poller.unregister(dealer_socket)
            subscriber_socket.close()
            dealer_socket.close()
            ctx.term()
            logger.debug(f'worker {worker_id:03d} has realsed all ressources')
        logger.debug(f'worker {worker_id:03d} end ...!')