import cv2
url = 'rtsp://admin:eksrufthgkr!@192.168.4.82:554/cam/realmonitor?channel=1&subtype=0'

cap = cv2.VideoCapture(url)
while True:
    ret, image = cap.read()
    cv2.imshow('stream', image)


cv2.destroyAllWindows()