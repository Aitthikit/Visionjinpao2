
import time
import numpy as np
from JinpaoVision import BoxDetect

# print = send to control
# pass = wait for ros for next state

#INIT
boxDetect = BoxDetect()
STATE = 'FIND_PATH_INIT'

def callback():
    current_time = time.time()

    # at INIT
    if STATE == 'FIND_PATH_INIT':
        boxDetect.findPath_INIT(current_time = current_time,
                                runtime = 5,
                                min_distance = 1,
                                max_distance = 1.9)
        STATE = 'FIND_PATH'
    elif STATE == 'FIND_PATH':
        if not (boxDetect.findPath(current_time = current_time)):
            pass # move to pick shelf






    # At pick shelf
    elif STATE == 'FIND_PICK_SHELF':
        error = boxDetect.findPickShelf(min_distance = 0.3,
                                        max_distance = 0.8)
        if not (error):
            STATE = 'FINETUNE_PICK_SHELF'

    elif STATE == 'FINETUNE_PICK_SHELF':
        x,y,z = boxDetect.finetune()
        print(x,y,z)

        # is in set point range
        if (np.abs(x) <= 20) and (np.abs(y) <= 100) and (z <= 0.1):
            STATE == 'INIT_PICK_SHELF'
    elif STATE == 'INIT_PICK_SHELF':
        Path = boxDetect.min_path
        Color = boxDetect.color
        path_idx = 0
        STATE = 'MOVING2PICK'

    elif STATE == 'MOVING2PICK':
        # z and x = 0
        if(Path[path_idx][1] == 0 and Path[path_idx][0] == 0):
            STATE = 'PICKING'
        # x = 0, z not 0
        elif(Path[path_idx][1] == 0 and Path[path_idx][0] != 0):
            print(Path[path_idx][0]) # go z
            pass # wait for move
        # x not 0, z = 0
        elif(Path[path_idx][1] != 0 and Path[path_idx][0] == 0):
            print(Path[path_idx][1]) # go x
            pass # wait for move
        else:
            print(Path[path_idx][1], Path[path_idx][0]) # go x and z
            pass # wait for move


    # If finished move
    elif STATE == 'INIT_PICKING':
        gripper = [0,0,0] # rgb
        STATE = 'PICKING'
    
    elif STATE == 'PICKING':
        if (path_idx < len(Path)):
            # if next = 00
            if (Path[path_idx + 1][1] == 0 and Path[path_idx + 1][0] == 0):
                #check gripper
                if Color[path_idx] == 'red':
                    gripper[0] = 30
                elif Color[path_idx] == 'green':
                    gripper[1] = 30
                elif Color[path_idx] == 'blue':
                    gripper[2] = 30
                
                path_idx += 1
                STATE = 'PICKING'
            else:
                #check gripper
                if Color[path_idx] == 'red':
                    gripper[0] = 30
                elif Color[path_idx] == 'green':
                    gripper[1] = 30
                elif Color[path_idx] == 'blue':
                    gripper[2] = 30
                
                path_idx += 1
                print(gripper) # send gripper
                pass # wait for gripper move (state = picking)
                
        # move all path
        else:
            pass # moveๆ to place






    # At place
    elif STATE == 'FIND_PLACE_SHELF':
        error = boxDetect.findPlaceShelf(min_distance = 0.3,
                                        max_distance = 0.8)
        print(error)
        if not (error):
            STATE = 'FINETUNE_PLACE_SHELF'
    
    elif STATE == 'FINETUNE_PLACE_SHELF':
        x,y,z = boxDetect.finetune()
        print(x,y,z)

        # is in set point range
        if (np.abs(x) <= 20) and (np.abs(y) <= 100) and (z <= 0.1):
            pass # go place


