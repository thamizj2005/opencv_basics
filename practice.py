# import cv2 as cv

# # cap=cv.VideoCapture('/home/thamizh/python_practice/opencv_basic/forest.mp4')
# cap=cv.VideoCapture(0)
# fps=cap.get(cv.CAP_PROP_FPS)
# delay=int(1000/fps)

# while True:
#     ret,frame=cap.read()
#     if not ret:
#         break
#     frame=cv.resize(frame,(1024,640))
#     gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
#     cv.imshow('VIDEO',frame)
#     cv.imshow('GRAY',gray)
#     if cv.waitKey(1) & 0xff==ord('q'):
#         break
# cap.release()
# cv.destroyAllWindows()



# import cv2 as cv

# img=cv.imread('./cars.png')
# cv.imshow('PIC',img)


# print(type(img))
# print(img.shape)

# re=cv.resize(img,(640,480))
# gray=cv.cvtColor(re,cv.COLOR_BGR2GRAY)
# print(gray.shape)


# cv.imshow('hi',gray)
# cv.waitKey(0)
# cv.destroyAllWindows()



# import cv2 as cv

# img=cv.imread('./cars.png')

# b,g,r=cv.split(img)
# cv.imshow('BLUE',b)
# cv.imshow('GREEN',g)
# cv.imshow('RED',r)
# print(b.shape)

# cv.waitKey(0)
# cv.destroyAllWindows()



# import cv2 as cv

# img=cv.imread('./cars.png')

# img[0,0]=[0,255,0]

# h,w,_=img.shape

# cx=w//2
# cy=h//2

# img[cy,cx]=[0,255,0]

# gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# cv.imshow('GRAY',gray)

# cv.waitKey(0)
# cv.destroyAllWindows()


# import cv2 as cv

# img = cv.imread("cars.png")

# h, w, _ = img.shape
# cx, cy = w//2, h//2

# cv.circle(img, (cx, cy), 5, (0,0,255),5)

# cv.imwrite("cars_with_center.jpg", img)



# import cv2 as cv

# cap = cv.VideoCapture(0)

# ret, frame = cap.read()
# if not ret:
#     print("Failed to read from camera")
#     cap.release()
#     exit()

# h, w, c = frame.shape

# fps = cap.get(cv.CAP_PROP_FPS)
# if fps == 0:
#     fps = 30  # default fallback

# delay = int(1000 / fps)

# print("Height:", h)
# print("Width:", w)
# print("Channels:", c)
# print("FPS:", fps)
# print("Delay (ms):", delay)

# cap.release()



import cv2 as cv

# capture the video

cap=cv.VideoCapture(0)

# verfiy the status

ret,frame=cap.read()
if not ret:
    print("Capture is not working")
    cap.release()
    exit()

# find the fps 

fps=cap.get(cv.CAP_PROP_FPS)
if fps==0:
    fps=30

# find widht and height

h,w,c= frame.shape


# define the codec

fourcc=cv.VideoWriter_fourcc(*'XVID')

out=cv.VideoWriter('output.avi',fourcc,fps,(w,h))

if not out.isOpened():
    print("Video write is  failed")
    cap.release()
    exit()

while True:
    ret,frame=cap.read()

    if not ret:
        break

    out.write(frame)
    cv.imshow("Recording",frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
out.release()
cv.destroyAllWindows()