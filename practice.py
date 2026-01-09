import cv2 as cv

# cap=cv.VideoCapture('/home/thamizh/python_practice/opencv_basic/forest.mp4')
cap=cv.VideoCapture(0)
fps=cap.get(cv.CAP_PROP_FPS)
delay=int(1000/fps)

while True:
    ret,frame=cap.read()
    if not ret:
        break
    frame=cv.resize(frame,(1024,640))
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    cv.imshow('VIDEO',frame)
    cv.imshow('GRAY',gray)
    if cv.waitKey(1) & 0xff==ord('q'):
        break
cap.release()
cv.destroyAllWindows()
