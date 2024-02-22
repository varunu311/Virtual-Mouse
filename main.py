import cv2
import numpy as np
import HandTracking as ht
import pyautogui
import threading

frame = None
frame_lock = threading.Lock()

def camera_thread(cap):
    global frame
    while cap.isOpened():
        ret, new_frame = cap.read()
        if not ret:
            break

        with frame_lock:
            frame = new_frame

def main():
    global frame
    width, height = 640, 480
    frameR = 180
    smoothening = 0
    prev_x, prev_y = 0, 0
    curr_x, curr_y = 0, 0

    cap = cv2.VideoCapture(0)
    cap.set(3, width)
    cap.set(4, height)

    detector = ht.handDetector(maxHands=1)
    screen_width, screen_height = pyautogui.size()

    threading.Thread(target=camera_thread, args=(cap,), daemon=True).start()

    pTime = 0

    def exponential_smoothing(a, b, alpha):
        return a * alpha + b * (1 - alpha)

    while True:
        with frame_lock:
            if frame is None:
                continue
            img = frame.copy()

        img = detector.findHands(img)
        lmlist, bbox = detector.findPosition(img)

        if len(lmlist) != 0:
            x1, y1 = lmlist[8][1:]
            x3 = np.interp(x1, (frameR, width - frameR), (0, screen_width))
            y3 = np.interp(y1, (frameR, height - frameR), (0, screen_height))

            fingers = detector.fingersUp()
            if fingers[1] == 1 and fingers[2] == 0:
                curr_x = exponential_smoothing(x3, prev_x, smoothening)
                curr_y = exponential_smoothing(y3, prev_y, smoothening)
                pyautogui.moveTo((screen_width - curr_x), (curr_y))
                prev_x, prev_y = curr_x, curr_y

            if fingers[1] == 1 and fingers[2] == 1:
                length, img, lineInfo = detector.findDistance(8, 12, img)
                if length < 40:
                    pyautogui.click()

        #cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
