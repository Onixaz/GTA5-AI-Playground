import numpy as np
from grabscreen import grab_screen
import cv2
import time
from getkeys import key_check
import os
from datetime import datetime

w = [1, 0, 0, 0, 0, 0, 0, 0, 0]
s = [0, 1, 0, 0, 0, 0, 0, 0, 0]
a = [0, 0, 1, 0, 0, 0, 0, 0, 0]
d = [0, 0, 0, 1, 0, 0, 0, 0, 0]
wa = [0, 0, 0, 0, 1, 0, 0, 0, 0]
wd = [0, 0, 0, 0, 0, 1, 0, 0, 0]
sa = [0, 0, 0, 0, 0, 0, 1, 0, 0]
sd = [0, 0, 0, 0, 0, 0, 0, 1, 0]
nk = [0, 0, 0, 0, 0, 0, 0, 0, 1]

t = datetime.now()
formatted_time = t.strftime('%y_%m_%d_%H_%M')


def keys_to_output(keys):
    '''
    Convert keys to a ...multi-hot... array
     0  1  2  3  4   5   6   7    8
    [W, S, A, D, WA, WD, SA, SD, NOKEY] boolean values.
    '''
    output = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    if 'W' in keys and 'A' in keys:
        output = wa
    elif 'W' in keys and 'D' in keys:
        output = wd
    elif 'S' in keys and 'A' in keys:
        output = sa
    elif 'S' in keys and 'D' in keys:
        output = sd
    elif 'W' in keys:
        output = w
    elif 'S' in keys:
        output = s
    elif 'A' in keys:
        output = a
    elif 'D' in keys:
        output = d
    else:
        output = nk
    return output


def preprocess_screen(screen):
    # conver to gray(from 3 to 1 channel)
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    # manually crop some irellavant information from the screen(in first person mode it crops most of the interior)
    screen = screen[100:400, 200:]
    # resize for the model
    screen = cv2.resize(screen, (224, 224))
    return screen


starting_value = 1

while True:
    file_name = 'E://CrossPlatform//Data//GTA5Training//training_data-{}.npy'.format(
        starting_value)
    if os.path.isfile(file_name):
        print(
            'File exists, creating new training_data.npy file')
        starting_value += 1
    else:
        print('Starting fresh file with training_data.npy number: ', starting_value)
        training_data = []

        break


def main(file_name, starting_value):

    file_name = file_name
    starting_value = starting_value

    print("Starting in....")
    for i in list(range(5))[::-1]:
        print(i+1)
        time.sleep(1)

    training_data = []

    paused = False
    while(True):
        if not paused:

            screen = grab_screen(region=(0, 40, 800, 640))
            last_time = time.time()

            screen = preprocess_screen(screen)

            keys = key_check()
            output = keys_to_output(keys)
            training_data.append([screen, output])
            last_time = time.time()

            cv2.imshow('window', cv2.resize(screen, (400, 320)))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

            if len(training_data) % 100 == 0:
                print(len(training_data))

                if len(training_data) == 500:
                    np.save(file_name, training_data)
                    print('SAVED')
                    training_data = []
                    starting_value += 1
                    file_name = 'E://CrossPlatform//Data//GTA5Training//training_data-{}.npy'.format(
                        starting_value)

        keys = key_check()
        if 'T' in keys:
            if paused:
                paused = False
                print('Unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
                time.sleep(1)


main(file_name, starting_value)
