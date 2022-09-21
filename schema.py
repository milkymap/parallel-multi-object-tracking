from enum import Enum
from typing import List, Tuple, Dict 
from pydantic import BaseModel

class ZMQModel(bytes, Enum):
    UPDATE_ROI:bytes=b'UPDATE_ROI'
    STOP_TRACKING:bytes=b'STOP_TRACKING'

    QUIT:bytes=b'QUIT'
    STREAM:bytes=b'STREAM'
    HANDSHAKE:bytes=b'HANDSHAKE'

    REFUSED:bytes=b'REFUSED'
    ACCEPTED:bytes=b'ACCEPTED'
    ESTABLISHED:bytes=b'ESTABLISHED'

    TRACKING_REQ:bytes=b'TRACKING_REQ'
    TRACKING_ACK:bytes=b'TRACKING_ACK'


