# Day 1 - OpenCV Practice Session
**Date:** January 9, 2026  
**Focus:** Video Capture, Image Processing Basics, and Video Recording

---

## 📚 Topics Covered

### 1. Video Capture and Display
**Objective:** Learn how to capture video from webcam and video files.

#### Key Concepts:
- **VideoCapture()**: Opens a video stream from webcam (index 0) or file path
- **cap.read()**: Reads a single frame, returns (ret, frame)
  - `ret`: Boolean indicating success
  - `frame`: The captured image as a numpy array
- **FPS (Frames Per Second)**: Retrieved using `cap.get(cv.CAP_PROP_FPS)`
- **Delay Calculation**: `delay = int(1000/fps)` for proper frame timing

#### Code Example:
```python
cap = cv.VideoCapture(0)  # 0 for webcam, or provide file path
fps = cap.get(cv.CAP_PROP_FPS)
delay = int(1000/fps)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv.imshow('VIDEO', frame)
    if cv.waitKey(1) & 0xff == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
```

---

### 2. Image Color Space Conversion
**Objective:** Convert images between different color spaces.

#### Key Concepts:
- **BGR to Grayscale**: `cv.cvtColor(frame, cv.COLOR_BGR2GRAY)`
- OpenCV uses **BGR** format (not RGB!)
- Grayscale images have only 1 channel vs 3 channels in BGR

#### Code Example:
```python
gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
cv.imshow('GRAY', gray)
```

---

### 3. Image Reading and Properties
**Objective:** Load images from disk and understand their properties.

#### Key Concepts:
- **imread()**: Reads an image from file
- **Image Shape**: `img.shape` returns (height, width, channels)
  - For BGR: (H, W, 3)
  - For Grayscale: (H, W)
- **Data Type**: Images are numpy arrays (`numpy.ndarray`)
- **Resizing**: `cv.resize(img, (width, height))`

#### Code Example:
```python
img = cv.imread('./cars.png')
print(type(img))  # <class 'numpy.ndarray'>
print(img.shape)  # e.g., (480, 640, 3)

resized = cv.resize(img, (640, 480))
gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
print(gray.shape)  # (480, 640) - no channel dimension
```

---

### 4. Channel Splitting
**Objective:** Separate BGR channels to understand color composition.

#### Key Concepts:
- **cv.split()**: Separates BGR image into 3 separate channels
- Each channel is a grayscale representation of that color's intensity
- Order: Blue, Green, Red (BGR)

#### Code Example:
```python
img = cv.imread('./cars.png')
b, g, r = cv.split(img)

cv.imshow('BLUE', b)
cv.imshow('GREEN', g)
cv.imshow('RED', r)
print(b.shape)  # Same H,W but single channel
```

---

### 5. Pixel Manipulation
**Objective:** Access and modify individual pixels.

#### Key Concepts:
- Pixels accessed as: `img[y, x]` or `img[row, col]`
- BGR format: `img[y, x] = [B, G, R]`
- Center calculation: `cx = w//2`, `cy = h//2`

#### Code Example:
```python
img = cv.imread('./cars.png')
h, w, _ = img.shape

# Set top-left pixel to green
img[0, 0] = [0, 255, 0]

# Set center pixel to green
cx, cy = w//2, h//2
img[cy, cx] = [0, 255, 0]
```

---

### 6. Drawing Shapes
**Objective:** Draw geometric shapes on images.

#### Key Concepts:
- **cv.circle()**: Draws a circle
  - Parameters: `(image, center, radius, color, thickness)`
  - Color in BGR format
  - Center as `(x, y)` tuple
- **cv.imwrite()**: Saves image to disk

#### Code Example:
```python
img = cv.imread("cars.png")
h, w, _ = img.shape
cx, cy = w//2, h//2

# Draw red circle at center
cv.circle(img, (cx, cy), 5, (0, 0, 255), 5)
cv.imwrite("cars_with_center.jpg", img)
```

---

### 7. Video Recording (Final Project)
**Objective:** Capture webcam feed and save to video file.

#### Key Concepts:
- **VideoWriter_fourcc()**: Defines codec for compression
  - Syntax: `cv.VideoWriter_fourcc(*'XVID')` - asterisk unpacks string
  - Common codecs: 'XVID', 'MJPG', 'mp4v'
- **VideoWriter()**: Creates video file writer
  - Parameters: `(filename, fourcc, fps, (width, height))`
  - **Important**: Dimensions must be (width, height), NOT (height, width)!
- **isOpened()**: Checks if VideoWriter initialized successfully
- **write()**: Writes a frame to the video file

#### Complete Working Code:
```python
import cv2 as cv

# Capture video from webcam
cap = cv.VideoCapture(0)

# Verify capture status
ret, frame = cap.read()
if not ret:
    print("Capture is not working")
    cap.release()
    exit()

# Get FPS
fps = cap.get(cv.CAP_PROP_FPS)
if fps == 0:
    fps = 30  # Default fallback

# Get frame dimensions
h, w, c = frame.shape

# Define codec (asterisk OUTSIDE quotes!)
fourcc = cv.VideoWriter_fourcc(*'XVID')

# Create VideoWriter (width, height order!)
out = cv.VideoWriter('output.avi', fourcc, fps, (w, h))

if not out.isOpened():
    print("Video writer failed")
    cap.release()
    exit()

# Recording loop
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    out.write(frame)
    cv.imshow("Recording", frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv.destroyAllWindows()
```

---

## 🐛 Common Errors & Fixes

### 1. **VideoWriter_fourcc() TypeError**
**Error:** `TypeError: VideoWriter_fourcc() missing required argument 'c2'`

**Cause:** Incorrect syntax - asterisk inside quotes
```python
# ❌ WRONG
fourcc = cv.VideoWriter_fourcc('*XVID')

# ✅ CORRECT
fourcc = cv.VideoWriter_fourcc(*'XVID')
```

**Explanation:** The asterisk unpacks 'XVID' into 4 separate characters: 'X', 'V', 'I', 'D'

### 2. **VideoWriter Dimensions**
**Common Mistake:** Swapping width and height
```python
# ❌ WRONG
out = cv.VideoWriter('output.avi', fourcc, fps, (h, w))

# ✅ CORRECT
out = cv.VideoWriter('output.avi', fourcc, fps, (w, h))
```

**Remember:** VideoWriter expects (width, height), but shape gives (height, width, channels)!

---

## 📊 Key Takeaways

1. **Image Shape Order**: (Height, Width, Channels)
2. **Coordinate Order**: (x, y) for positions, but img[row, col] = img[y, x]
3. **Color Format**: OpenCV uses **BGR**, not RGB
4. **Video Dimensions**: VideoWriter uses (width, height)
5. **FPS Handling**: Always check if FPS is 0 and provide fallback
6. **Resource Cleanup**: Always call `release()` and `destroyAllWindows()`

---

## 🎯 Skills Achieved Today

- ✅ Captured video from webcam
- ✅ Processed video frames in real-time
- ✅ Converted color spaces (BGR ↔ Grayscale)
- ✅ Manipulated individual pixels
- ✅ Split and analyzed color channels
- ✅ Drew shapes on images
- ✅ Recorded webcam to video file
- ✅ Debugged VideoWriter errors

---

## 📝 Notes

- **Qt Warning**: The "wayland" plugin warning can be safely ignored - it's related to the GUI backend and doesn't affect functionality
- **Output File**: The final code creates `output.avi` in the current directory
- **Press 'q'**: Standard convention to quit OpenCV windows

---

## 🚀 Next Steps

1. Explore more drawing functions (rectangle, line, text)
2. Learn image filtering and blurring
3. Practice edge detection
4. Try different video codecs
5. Experiment with real-time video effects
