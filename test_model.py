from directkeys import PressKey, ReleaseKey, W, A, S, D
import numpy as np
from keras.models import load_model
import random
import cv2

from grabscreen import grab_screen
import time
from getkeys import key_check


def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)


def left():
    if random.randrange(0, 3) == 1:
        PressKey(W)
    else:
        ReleaseKey(W)
    PressKey(A)
    ReleaseKey(S)
    ReleaseKey(D)
# ReleaseKey(S)


def right():
    if random.randrange(0, 3) == 1:
        PressKey(W)
    else:
        ReleaseKey(W)
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)


def reverse():
    PressKey(S)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)


def forward_left():
    PressKey(W)
    PressKey(A)
    ReleaseKey(D)
    ReleaseKey(S)


def forward_right():
    PressKey(W)
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)


def reverse_left():
    PressKey(S)
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)


def reverse_right():
    PressKey(S)
    PressKey(D)
    ReleaseKey(W)
    ReleaseKey(A)


def no_keys():
    if random.randrange(0, 3) == 1:
        PressKey(W)
    else:
        ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(D)


model = load_model('model_new_2.h5')
paused = False

print("Starting in... ")
for i in list(range(5))[::-1]:
    print(i+1)
    time.sleep(1)

while True:
    if not paused:
        last_time = time.time()

        screen = grab_screen(region=(0, 40, 800, 600))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        screen = screen[100:400, 200:]
        screen_for_prediction = cv2.resize(
            screen, (224, 224))
        screen_for_prediction = cv2.GaussianBlur(
            screen_for_prediction, (3, 3), 0)
        screen_for_prediction = screen_for_prediction.reshape(1,
                                                              224, 224, 1)
        prediction = model.predict(screen_for_prediction)[0]
        mode_choice = np.argmax(prediction)
        if mode_choice == 0:
            straight()
            choice_picked = 'straight'
        elif mode_choice == 1:
            reverse()
            choice_picked = 'reverse'
        elif mode_choice == 2:
            left()
            choice_picked = 'left'
        elif mode_choice == 3:
            right()
            choice_picked = 'right'
        elif mode_choice == 4:
            forward_left()
            choice_picked = 'forward+left'
        elif mode_choice == 5:
            forward_right()
            choice_picked = 'forward+right'
        elif mode_choice == 6:
            reverse_left()
            choice_picked = 'reverse+left'
        elif mode_choice == 7:
            reverse_right()
            choice_picked = 'reverse+right'
        elif mode_choice == 8:
            no_keys()
            choice_picked = 'nokeys'

        ###Print FPS ####
        print("Fps: {} Prediction: {}".format(
            1 / (time.time() - last_time), choice_picked))
        cv2.imshow("Screen", cv2.resize(screen, (400, 200)))
        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
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
            ReleaseKey(A)
            ReleaseKey(W)
            ReleaseKey(D)
            time.sleep(1)
