import click 
import uuid 

import cv2 
import numpy as np 
import operator as op 
import itertools as it, functools as ft 

import zmq 
import time 
import threading 
import multiprocessing as mp 

from rich.progress import track 
from libraries.strategies import * 
from libraries.log import logger 

@click.group(chain=False)
@click.pass_context
def group_cli(clx):
    subcommand = clx.invoked_subcommand 
    if subcommand is not None:
        logger.debug(f'{subcommand} was called')
    else:
        logger.debug('use --help to see available commands')

def create_window(window_name, width, height):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)

@group_cli.command()
@click.option('--path2video', type=click.Path(exists=True))
@click.option('--tracker_type', help='type of tracker', type=click.Choice(['KCF', 'CSRT', 'MIL']))
@click.option('--window_name', type=str, default='display')
@click.option('--window_size', type=click.Tuple([int, int]), default=(800, 800))
def sequential_pipeline(path2video, tracker_type, window_name, window_size):
    logger.debug('sequential-pipeline will start')
    
    main_window = f'{window_name}-000'
    alarm_window = f'{window_name}-001'
    
    create_window(main_window, *window_size)
    create_window(alarm_window, *window_size)

    template = f'Tracker{tracker_type}_create'
    tracker_builder = op.attrgetter(template)(cv2)
    
    colors = []
    points_acc = []
    trackers_acc = []
    subtractors_acc = []
    selected_regions = None
    
    black_screen = np.zeros((*window_size, 3), dtype=np.uint8)
    grabber = cv2.VideoCapture(path2video)
    keep_grabbing = True 
    while keep_grabbing:
        key_code = cv2.waitKey(25) & 0xFF  # listening keyboard events for 25ms  
        read_status, bgr_frame = grabber.read()
        keep_grabbing = read_status and key_code != 27  #   
        if keep_grabbing:    
            resized_bgr_frame = cv2.resize(bgr_frame, window_size)
            
            if selected_regions is not None:
                temporaries_regions = []
                for idx, (tracker, inital_roi, subtractor) in enumerate(zip(trackers_acc, points_acc, subtractors_acc)):
                    tracking_status, predicted_roi = tracker.update(resized_bgr_frame)
                    if tracking_status:
                        a, b, c, d = inital_roi
                        x, y, w, h = list(map(int, predicted_roi))
                        #binary_mask = subtractor.apply(resized_bgr_frame[y:y+h, x:x+w])
                        temporaries_regions.append((x, y, w, h))
                        cv2.rectangle(resized_bgr_frame, (x, y), (x + w, y + h), colors[idx], 3)
                        cv2.line(resized_bgr_frame, (a + c // 2, b + d // 2), (x + w // 2, y + h // 2), (0, 0, 255), 3)
                    else:
                        colors.pop(idx)
                        points_acc.pop(idx)
                        trackers_acc.pop(idx)
                        subtractors_acc.pop(idx)
                        selected_regions.pop(idx)
                # end loop over trackers 

                assert len(temporaries_regions) == len(selected_regions)
                for box0, box1 in zip(points_acc, selected_regions):
                    value = int(distance(box0, box1))
                    if value > 10:
                        message = f'object is moving => distance : {value:03d}'
                        logger.error(message)
                        display_message(black_screen, message, (0, 0, 255))
                        cv2.imshow(alarm_window, black_screen)
                        black_screen *= 0 

                selected_regions = temporaries_regions

            if key_code == 32: 
                selected_regions = cv2.selectROIs(
                    windowName=main_window, 
                    img=resized_bgr_frame, 
                    fromCenter=False, 
                    showCrosshair=True
                )
                nb_regions = len(selected_regions)
                colors = list(map(tuple, np.random.randint(0, 255, size=(nb_regions, 3)).tolist()))
                for roi in track(selected_regions, 'tracker initialization'):
                    tracker = tracker_builder()
                    tracker.init(resized_bgr_frame, roi)
                    subtractor = cv2.createBackgroundSubtractorKNN()
                    points_acc.append(roi)
                    trackers_acc.append(tracker)
                    subtractors_acc.append(subtractor)

            cv2.imshow(main_window, resized_bgr_frame)
    
    cv2.destroyAllWindows()

@group_cli.command()
@click.option('--path2video', type=click.Path(exists=True))
@click.option('--tracker_type', help='type of tracker', type=click.Choice(['KCF', 'CSRT', 'MIL']))
@click.option('--window_name', type=str, default='display')
@click.option('--window_size', type=click.Tuple([int, int]), default=(800, 800))
@click.option('--nb_workers', help='number of worker(multi-trackers)', type=int, default=4)
@click.option('--worker_limit', help='max number of trackers per worker', type=int, default=8)
@click.option('--timeout', help='zeromq timeout', type=int, default=5000)
@click.option('--security_distance', type=float, default=10)
def distributed_pipeline(path2video, tracker_type, window_name, window_size, nb_workers, worker_limit, timeout, security_distance):
    ZEROMQ_INIT = 0 

    try:

        max_nb_objects = worker_limit * nb_workers
        np_colors = np.random.randint(0, 256, size=(nb_workers, 3)).tolist()
        colors = list(map(tuple, np_colors))

        # intialize zeromq socket 
        ctx = zmq.Context()
        router_socket:zmq.Socket = ctx.socket(zmq.ROUTER)
        publisher_socket:zmq.Socket = ctx.socket(zmq.PUB)
        
        router_address = 'ipc://router.ipc'
        publisher_address = 'ipc://publisher.ipc'

        router_socket.bind(router_address)
        publisher_socket.bind(publisher_address)

        ZEROMQ_INIT = 1 
        # multiprocess syncrhonizer 
        flag = mp.Value('i', 0)
        readyness = mp.Event()
        condition = mp.Condition()
        message_queue = mp.Queue()
        

        # create workers  
        processes_acc = []
        for worker_id in range(nb_workers):
            process_ = mp.Process(
                target=worker, 
                kwargs={
                    'worker_id': worker_id,
                    'router_address': router_address, 
                    'publisher_address': publisher_address, 
                    'timeout': timeout, 
                    'tracker_type': tracker_type, 
                    'readyness': readyness, 
                    'condition': condition, 
                    'flag': flag, 
                    'message_queue': message_queue 
                }
            )
            processes_acc.append(process_)
            processes_acc[-1].start()
        
        logger.debug('server is waiting for workers to be ready')
        time.sleep(1)  # wait 1s for worker to create their sockets 
        readyness.set()  # send signal to workers 
        
        # accept connection
        publisher_socket.send_multipart([ZMQModel.HANDSHAKE, b''])

        limit = 100
        counter = 0
        nb_connections = 0 
        while nb_connections < nb_workers and counter < timeout:
            router_poller_status = router_socket.poll(limit)
            counter += limit 
            if router_poller_status == zmq.POLLIN:
                worker_id, _, message_from_worker = router_socket.recv_multipart()
                message = pickle.loads(message_from_worker)
                if message['type'] == ZMQModel.HANDSHAKE:
                    router_socket.send_multipart([worker_id, b'', ZMQModel.ACCEPTED])
                    nb_connections = nb_connections + 1
                    logger.success(f'server has accepted a new connection from {worker_id}')
        # end loop connections ...!
        assert nb_connections == nb_workers
        logger.success('all workers are connected')
        
        # vision loop
        worker_weights = np.zeros(nb_workers)
        map_region_id2description = {}

        create_window(window_name, *window_size)
        grabber = cv2.VideoCapture(path2video)
        keep_grabbing = True 
        while keep_grabbing:
            key_code = cv2.waitKey(25) & 0xFF
            reading_status, bgr_image = grabber.read()
            keep_grabbing = reading_status and key_code != 27 
            logger.debug(f'worker states => {worker_weights}')
            if keep_grabbing:
                bgr_image = cv2.resize(bgr_image, window_size)  
                if len(map_region_id2description) > 0:      
                    condition.acquire()
                    publisher_socket.send(ZMQModel.STREAM, flags=zmq.SNDMORE)
                    publisher_socket.send_pyobj(bgr_image)  # broadcast bgr_image to all workers  
                    condition.wait_for(lambda: flag.value == nb_workers, timeout=10)  # wait 10s for workers to terminate the tracking 
                    
                    assert flag.value == nb_workers
                    while message_queue.qsize() > 0:
                        tracking_response = message_queue.get()
                        if tracking_response['type'] == ZMQModel.UPDATE_ROI:
                            map_region_id2description[ tracking_response['data']['region_id'] ]['crr_coordinates'] = tracking_response['data']['coordinates']
                        else:
                            worker_weights[ map_region_id2description[ tracking_response['data']['region_id'] ]['worker_id'] ] -= 1
                            del map_region_id2description[ tracking_response['data']['region_id'] ]

                    for region_id, description in map_region_id2description.items():
                        crr_status = map_region_id2description[region_id]['status']
                        if crr_status > security_distance:
                            # send notification to remote user 
                            logger.warning(f'{region_id} is moving : abnormal motion was detected distance => {crr_status:03d}')

                        a, b, c, d = description['ini_position']
                        x, y, w, h = description['crr_coordinates']

                        cv2.rectangle(bgr_image, (x, y), (x + w, y + h), description['color'], 3)
                        cv2.line(bgr_image, (a + c // 2, b + d // 2), (x + w // 2, y + h // 2), (0, 0, 255), 5)

                        value = int(distance((a, b, c, d), (x, y, w, h)))
                        map_region_id2description[region_id]['status'] = value 
                            
                    with flag.get_lock():  # reset the flag for future messages 
                        flag.value = 0

                if key_code == 32:
                    if len(map_region_id2description) == max_nb_objects:
                        logger.warning('max number of objects to track was reached')
                    else:
                        regions = cv2.selectROIs(window_name, bgr_image, fromCenter=False, showCrosshair=True)
                        if len(regions) > 0: 
                            for coordinates in regions:
                                idx = np.argmin(worker_weights)
                                if worker_weights[idx] < worker_limit:
                                    region_id = str(uuid.uuid4())
                                    worker_address = f'{idx:03d}'
                                    router_socket.send_multipart([worker_address.encode(), b''], flags=zmq.SNDMORE)
                                    router_socket.send_pyobj({
                                        'type': ZMQModel.TRACKING_REQ, 
                                        'data': {
                                            'bgr_image': bgr_image,
                                            'region_id': region_id, 
                                            'coordinates': coordinates
                                        }
                                    })

                                    router_poller_status = router_socket.poll(5000)
                                    if router_poller_status == zmq.POLLIN: 
                                        worker_id, _, message_from_worker = router_socket.recv_multipart()
                                        assert worker_id == worker_address.encode()
                                        message = pickle.loads(message_from_worker)
                                        if message['type'] == ZMQModel.TRACKING_ACK:
                                            if message['data'] == ZMQModel.ACCEPTED:
                                                logger.debug(f'{worker_id} has confirmed the tracking')
                                                description = {
                                                    'color': colors[idx], 
                                                    'status': 0,   # 0 : static | 1 : moved 
                                                    'worker_id': idx, 
                                                    'ini_position': coordinates, 
                                                    'crr_coordinates': coordinates
                                                } 
                                                map_region_id2description[region_id] = description
                                                worker_weights[idx] += 1
                                    # end if router pollin status 
                            # end loop over selected regions 
                        # end if len region > 0 
                # end region selection     
                cv2.imshow(window_name, bgr_image)
        # end ...!

    except KeyboardInterrupt:
        pass 
    except Exception as e:
        logger.error(e)
    finally:
        if ZEROMQ_INIT:
            publisher_socket.send_multipart([ZMQModel.QUIT, b''])
            for process_ in processes_acc:
                process_.join()

            publisher_socket.close()
            router_socket.close()
            ctx.term()
            logger.debug('server has removed all zmq ressources')
        logger.debug('server end ...!')

if __name__ == '__main__':
    logger.debug(' ... video object tracking ... ')
    group_cli(obj={})