
import time
import numpy as np
from JinpaoVision import BoxDetect

boxDetect = BoxDetect()
STATE = 'FIND_PLACE_SHELF'

while True:

    current_time = time.time()
    if STATE == 'INIT':
        print(STATE)
        boxDetect.findPath_INIT(current_time = current_time,
                                runtime = 5,
                                min_distance = 1,
                                max_distance = 1.9)
        STATE = 'FIND_PATH'
        print(STATE)

    elif STATE == 'FIND_PATH':
        
        if not (boxDetect.findPath(current_time = current_time)):
            STATE = 'FIND_PICK_SHELF'
            print(STATE)

    elif STATE == 'FIND_PICK_SHELF':
        error = boxDetect.findPickShelf(min_distance = 0.3,
                                        max_distance = 0.8)
        print(error)
        if not (error):
            STATE = 'FINETUNE_PICK_SHELF'
            print(STATE)
    elif STATE == 'FINETUNE_PICK_SHELF':
        x,y,z = boxDetect.finetune()
        print(x,y,z)
        if (np.abs(x) <= 20) and (np.abs(y) <= 100) and (z <= 0.1):
            print("fin")

    elif STATE == 'FIND_PLACE_SHELF':
        error = boxDetect.findPlaceShelf(min_distance = 0.3,
                                        max_distance = 0.8)
        print(error)
        if not (error):
            STATE = 'FINETUNE_PLACE_SHELF'
    
    elif STATE == 'FINETUNE_PLACE_SHELF':
        x,y,z = boxDetect.finetune()
        print(x,y,z)
        if (np.abs(x) <= 20) and (np.abs(y) <= 100) and (z <= 0.1):
            print("fin")


