# OpenCV Daily Revision Reference Guide

## Image Operations

| Function | Syntax | Parameters | Description | Example |
|----------|--------|------------|-------------|---------|
| **imread()** | `cv2.imread(path, flag)` | `path`: Image file path<br>`flag`: Read mode | Loads an image from file | `img = cv2.imread('image.jpg', cv2.IMREAD_COLOR)` |
| **imshow()** | `cv2.imshow(window_name, image)` | `window_name`: Window title<br>`image`: Image array | Displays image in window | `cv2.imshow('Image', img)` |
| **imwrite()** | `cv2.imwrite(filename, image)` | `filename`: Save path<br>`image`: Image to save | Saves image to file | `cv2.imwrite('output.jpg', img)` |
| **waitKey()** | `cv2.waitKey(delay)` | `delay`: Wait time in ms (0 = infinite) | Waits for key press | `cv2.waitKey(0)` or `cv2.waitKey(1)` |
| **destroyAllWindows()** | `cv2.destroyAllWindows()` | None | Closes all OpenCV windows | `cv2.destroyAllWindows()` |
| **destroyWindow()** | `cv2.destroyWindow(window_name)` | `window_name`: Specific window | Closes specific window | `cv2.destroyWindow('Image')` |

### Read Flags for imread()
- `cv2.IMREAD_COLOR` or `1` - Color image (default)
- `cv2.IMREAD_GRAYSCALE` or `0` - Grayscale image
- `cv2.IMREAD_UNCHANGED` or `-1` - Image with alpha channel

---

## Video Capture Operations

| Function | Syntax | Parameters | Description | Example |
|----------|--------|------------|-------------|---------|
| **VideoCapture()** | `cv2.VideoCapture(source)` | `source`: Camera index or video path | Opens video file or camera | `cap = cv2.VideoCapture(0)` |
| **read()** | `cap.read()` | None | Reads next frame | `ret, frame = cap.read()` |
| **release()** | `cap.release()` | None | Releases video capture object | `cap.release()` |
| **isOpened()** | `cap.isOpened()` | None | Checks if capture is initialized | `if cap.isOpened():` |
| **get()** | `cap.get(propId)` | `propId`: Property identifier | Gets property value | `fps = cap.get(cv2.CAP_PROP_FPS)` |
| **set()** | `cap.set(propId, value)` | `propId`: Property ID<br>`value`: New value | Sets property value | `cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)` |

### Common Video Properties
- `cv2.CAP_PROP_FPS` - Frame rate
- `cv2.CAP_PROP_FRAME_WIDTH` - Frame width
- `cv2.CAP_PROP_FRAME_HEIGHT` - Frame height
- `cv2.CAP_PROP_FRAME_COUNT` - Total frame count
- `cv2.CAP_PROP_POS_FRAMES` - Current frame position
- `cv2.CAP_PROP_POS_MSEC` - Current position in milliseconds

---

## Video Writing Operations

| Function | Syntax | Parameters | Description | Example |
|----------|--------|------------|-------------|---------|
| **VideoWriter()** | `cv2.VideoWriter(filename, fourcc, fps, frameSize)` | `filename`: Output path<br>`fourcc`: Codec<br>`fps`: Frame rate<br>`frameSize`: (width, height) | Creates video writer object | `out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))` |
| **VideoWriter_fourcc()** | `cv2.VideoWriter_fourcc(*'CODE')` | Codec code (4 characters) | Defines video codec | `fourcc = cv2.VideoWriter_fourcc(*'XVID')` |
| **write()** | `out.write(frame)` | `frame`: Frame to write | Writes frame to video | `out.write(frame)` |

### Common Codecs
- `*'XVID'` - XVID codec (.avi)
- `*'MP4V'` - MPEG-4 codec (.mp4)
- `*'X264'` - H.264 codec
- `*'MJPG'` - Motion JPEG (.avi)

---

## Color Space Conversion

| Function | Syntax | Parameters | Description | Example |
|----------|--------|------------|-------------|---------|
| **cvtColor()** | `cv2.cvtColor(src, code)` | `src`: Input image<br>`code`: Conversion code | Converts color space | `gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` |

### Common Conversion Codes
- `cv2.COLOR_BGR2GRAY` - BGR to Grayscale
- `cv2.COLOR_BGR2RGB` - BGR to RGB
- `cv2.COLOR_BGR2HSV` - BGR to HSV
- `cv2.COLOR_GRAY2BGR` - Grayscale to BGR
- `cv2.COLOR_HSV2BGR` - HSV to BGR

---

## Image Transformations

| Function | Syntax | Parameters | Description | Example |
|----------|--------|------------|-------------|---------|
| **resize()** | `cv2.resize(src, dsize, fx, fy, interpolation)` | `src`: Input image<br>`dsize`: Output size<br>`fx, fy`: Scale factors<br>`interpolation`: Method | Resizes image | `resized = cv2.resize(img, (300, 300))` |
| **flip()** | `cv2.flip(src, flipCode)` | `src`: Input image<br>`flipCode`: Flip direction | Flips image | `flipped = cv2.flip(img, 1)` |
| **rotate()** | `cv2.rotate(src, rotateCode)` | `src`: Input image<br>`rotateCode`: Rotation type | Rotates image by 90° | `rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)` |
| **getRotationMatrix2D()** | `cv2.getRotationMatrix2D(center, angle, scale)` | `center`: Rotation center<br>`angle`: Degrees<br>`scale`: Scale factor | Gets rotation matrix | `M = cv2.getRotationMatrix2D((50,50), 45, 1.0)` |
| **warpAffine()** | `cv2.warpAffine(src, M, dsize)` | `src`: Input image<br>`M`: Transformation matrix<br>`dsize`: Output size | Applies affine transformation | `rotated = cv2.warpAffine(img, M, (300,300))` |

### Flip Codes
- `0` - Flip vertically (around x-axis)
- `1` - Flip horizontally (around y-axis)
- `-1` - Flip both vertically and horizontally

### Interpolation Methods
- `cv2.INTER_LINEAR` - Bilinear (default)
- `cv2.INTER_NEAREST` - Nearest neighbor
- `cv2.INTER_CUBIC` - Bicubic
- `cv2.INTER_AREA` - Resampling using pixel area

---

## Drawing Functions

| Function | Syntax | Parameters | Description | Example |
|----------|--------|------------|-------------|---------|
| **line()** | `cv2.line(img, pt1, pt2, color, thickness)` | `img`: Image<br>`pt1, pt2`: Start/end points<br>`color`: RGB/BGR tuple<br>`thickness`: Line width | Draws line | `cv2.line(img, (0,0), (100,100), (255,0,0), 2)` |
| **rectangle()** | `cv2.rectangle(img, pt1, pt2, color, thickness)` | `img`: Image<br>`pt1, pt2`: Opposite corners<br>`color`: Color<br>`thickness`: Width (-1 = filled) | Draws rectangle | `cv2.rectangle(img, (10,10), (200,200), (0,255,0), 3)` |
| **circle()** | `cv2.circle(img, center, radius, color, thickness)` | `img`: Image<br>`center`: Center point<br>`radius`: Circle radius<br>`color`: Color<br>`thickness`: Width | Draws circle | `cv2.circle(img, (100,100), 50, (0,0,255), -1)` |
| **ellipse()** | `cv2.ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness)` | Multiple parameters for ellipse shape | Draws ellipse | `cv2.ellipse(img, (100,100), (80,40), 0, 0, 360, (255,0,0), 2)` |
| **putText()** | `cv2.putText(img, text, org, font, fontScale, color, thickness)` | `img`: Image<br>`text`: Text string<br>`org`: Bottom-left position<br>`font`: Font type<br>`fontScale`: Size<br>`color`: Color<br>`thickness`: Width | Adds text to image | `cv2.putText(img, 'Hello', (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)` |
| **polylines()** | `cv2.polylines(img, pts, isClosed, color, thickness)` | `img`: Image<br>`pts`: Array of points<br>`isClosed`: Close polygon<br>`color`: Color<br>`thickness`: Width | Draws polygon | `cv2.polylines(img, [pts], True, (0,255,0), 2)` |

### Common Fonts
- `cv2.FONT_HERSHEY_SIMPLEX`
- `cv2.FONT_HERSHEY_PLAIN`
- `cv2.FONT_HERSHEY_COMPLEX`
- `cv2.FONT_HERSHEY_TRIPLEX`

---

## Image Processing

| Function | Syntax | Parameters | Description | Example |
|----------|--------|------------|-------------|---------|
| **GaussianBlur()** | `cv2.GaussianBlur(src, ksize, sigmaX)` | `src`: Input image<br>`ksize`: Kernel size (odd)<br>`sigmaX`: Std deviation | Applies Gaussian blur | `blurred = cv2.GaussianBlur(img, (5,5), 0)` |
| **medianBlur()** | `cv2.medianBlur(src, ksize)` | `src`: Input image<br>`ksize`: Kernel size (odd) | Applies median blur | `blurred = cv2.medianBlur(img, 5)` |
| **bilateralFilter()** | `cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace)` | `src`: Input<br>`d`: Diameter<br>`sigmaColor`: Color sigma<br>`sigmaSpace`: Space sigma | Edge-preserving blur | `filtered = cv2.bilateralFilter(img, 9, 75, 75)` |
| **Canny()** | `cv2.Canny(image, threshold1, threshold2)` | `image`: Input<br>`threshold1`: Lower threshold<br>`threshold2`: Upper threshold | Edge detection | `edges = cv2.Canny(img, 100, 200)` |
| **threshold()** | `cv2.threshold(src, thresh, maxval, type)` | `src`: Input<br>`thresh`: Threshold value<br>`maxval`: Max value<br>`type`: Threshold type | Binary thresholding | `ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)` |
| **adaptiveThreshold()** | `cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C)` | Multiple adaptive parameters | Adaptive thresholding | `thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)` |
| **erode()** | `cv2.erode(src, kernel, iterations)` | `src`: Input<br>`kernel`: Structuring element<br>`iterations`: Times to apply | Morphological erosion | `eroded = cv2.erode(img, kernel, iterations=1)` |
| **dilate()** | `cv2.dilate(src, kernel, iterations)` | `src`: Input<br>`kernel`: Structuring element<br>`iterations`: Times to apply | Morphological dilation | `dilated = cv2.dilate(img, kernel, iterations=1)` |
| **morphologyEx()** | `cv2.morphologyEx(src, op, kernel)` | `src`: Input<br>`op`: Operation type<br>`kernel`: Structuring element | Advanced morphology | `opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)` |

### Threshold Types
- `cv2.THRESH_BINARY` - Binary threshold
- `cv2.THRESH_BINARY_INV` - Inverted binary
- `cv2.THRESH_TRUNC` - Truncate
- `cv2.THRESH_TOZERO` - To zero
- `cv2.THRESH_OTSU` - Otsu's method

### Morphological Operations
- `cv2.MORPH_OPEN` - Opening (erosion → dilation)
- `cv2.MORPH_CLOSE` - Closing (dilation → erosion)
- `cv2.MORPH_GRADIENT` - Morphological gradient
- `cv2.MORPH_TOPHAT` - Top hat
- `cv2.MORPH_BLACKHAT` - Black hat

---

## Contour Operations

| Function | Syntax | Parameters | Description | Example |
|----------|--------|------------|-------------|---------|
| **findContours()** | `cv2.findContours(image, mode, method)` | `image`: Binary image<br>`mode`: Retrieval mode<br>`method`: Approximation | Finds contours | `contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)` |
| **drawContours()** | `cv2.drawContours(image, contours, contourIdx, color, thickness)` | `image`: Image to draw on<br>`contours`: Contour list<br>`contourIdx`: Index (-1 = all)<br>`color`: Color<br>`thickness`: Width | Draws contours | `cv2.drawContours(img, contours, -1, (0,255,0), 2)` |
| **contourArea()** | `cv2.contourArea(contour)` | `contour`: Single contour | Calculates contour area | `area = cv2.contourArea(cnt)` |
| **arcLength()** | `cv2.arcLength(contour, closed)` | `contour`: Contour<br>`closed`: Is closed | Calculates perimeter | `perimeter = cv2.arcLength(cnt, True)` |
| **approxPolyDP()** | `cv2.approxPolyDP(contour, epsilon, closed)` | `contour`: Input contour<br>`epsilon`: Approximation accuracy<br>`closed`: Is closed | Approximates contour | `approx = cv2.approxPolyDP(cnt, 0.01*perimeter, True)` |
| **boundingRect()** | `cv2.boundingRect(contour)` | `contour`: Input contour | Gets bounding rectangle | `x, y, w, h = cv2.boundingRect(cnt)` |
| **minEnclosingCircle()** | `cv2.minEnclosingCircle(contour)` | `contour`: Input contour | Gets minimum enclosing circle | `(x,y), radius = cv2.minEnclosingCircle(cnt)` |
| **moments()** | `cv2.moments(contour)` | `contour`: Input contour | Calculates contour moments | `M = cv2.moments(cnt)` |

### Contour Retrieval Modes
- `cv2.RETR_EXTERNAL` - Only extreme outer contours
- `cv2.RETR_LIST` - All contours (no hierarchy)
- `cv2.RETR_TREE` - All contours with full hierarchy
- `cv2.RETR_CCOMP` - Two-level hierarchy

### Contour Approximation Methods
- `cv2.CHAIN_APPROX_NONE` - All points
- `cv2.CHAIN_APPROX_SIMPLE` - Compressed contours

---

## Feature Detection

| Function | Syntax | Parameters | Description | Example |
|----------|--------|------------|-------------|---------|
| **goodFeaturesToTrack()** | `cv2.goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance)` | `image`: Grayscale<br>`maxCorners`: Max number<br>`qualityLevel`: Quality<br>`minDistance`: Min distance | Detects corners | `corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)` |
| **HoughLines()** | `cv2.HoughLines(image, rho, theta, threshold)` | `image`: Binary<br>`rho`: Distance resolution<br>`theta`: Angle resolution<br>`threshold`: Threshold | Detects lines | `lines = cv2.HoughLines(edges, 1, np.pi/180, 200)` |
| **HoughCircles()** | `cv2.HoughCircles(image, method, dp, minDist)` | Multiple parameters for circle detection | Detects circles | `circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20)` |

---

## Image Arithmetic

| Function | Syntax | Parameters | Description | Example |
|----------|--------|------------|-------------|---------|
| **add()** | `cv2.add(src1, src2)` | `src1, src2`: Input images | Adds images (saturated) | `result = cv2.add(img1, img2)` |
| **subtract()** | `cv2.subtract(src1, src2)` | `src1, src2`: Input images | Subtracts images | `result = cv2.subtract(img1, img2)` |
| **multiply()** | `cv2.multiply(src1, src2)` | `src1, src2`: Input images | Multiplies images | `result = cv2.multiply(img1, img2)` |
| **divide()** | `cv2.divide(src1, src2)` | `src1, src2`: Input images | Divides images | `result = cv2.divide(img1, img2)` |
| **addWeighted()** | `cv2.addWeighted(src1, alpha, src2, beta, gamma)` | `src1, src2`: Images<br>`alpha, beta`: Weights<br>`gamma`: Scalar | Weighted image addition | `blended = cv2.addWeighted(img1, 0.7, img2, 0.3, 0)` |
| **bitwise_and()** | `cv2.bitwise_and(src1, src2, mask)` | `src1, src2`: Images<br>`mask`: Optional mask | Bitwise AND operation | `result = cv2.bitwise_and(img1, img2)` |
| **bitwise_or()** | `cv2.bitwise_or(src1, src2)` | `src1, src2`: Input images | Bitwise OR operation | `result = cv2.bitwise_or(img1, img2)` |
| **bitwise_not()** | `cv2.bitwise_not(src)` | `src`: Input image | Bitwise NOT operation | `result = cv2.bitwise_not(img)` |

---

## Utility Functions

| Function | Syntax | Parameters | Description | Example |
|----------|--------|------------|-------------|---------|
| **split()** | `cv2.split(image)` | `image`: Multi-channel image | Splits image into channels | `b, g, r = cv2.split(img)` |
| **merge()** | `cv2.merge([ch1, ch2, ch3])` | List of channels | Merges channels into image | `img = cv2.merge([b, g, r])` |
| **getStructuringElement()** | `cv2.getStructuringElement(shape, ksize)` | `shape`: Element shape<br>`ksize`: Kernel size | Creates structuring element | `kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))` |
| **namedWindow()** | `cv2.namedWindow(window_name, flags)` | `window_name`: Name<br>`flags`: Window properties | Creates named window | `cv2.namedWindow('Image', cv2.WINDOW_NORMAL)` |

### Structuring Element Shapes
- `cv2.MORPH_RECT` - Rectangular
- `cv2.MORPH_ELLIPSE` - Elliptical
- `cv2.MORPH_CROSS` - Cross-shaped

### Window Flags
- `cv2.WINDOW_NORMAL` - Resizable window
- `cv2.WINDOW_AUTOSIZE` - Auto-sized to image

---

## Common Code Snippets

### Basic Image Display
```python
import cv2

img = cv2.imread('image.jpg')
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Video Capture Loop
```python
import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Get Video Properties
```python
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
```

### Save Video
```python
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

while True:
    ret, frame = cap.read()
    if ret:
        out.write(frame)
    else:
        break

out.release()
```

### Region of Interest (ROI)
```python
roi = img[100:300, 150:350]  # [y1:y2, x1:x2]
img[400:600, 200:400] = roi
```

### Mouse Callback
```python
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at: ({x}, {y})")

cv2.namedWindow('Image')
cv2.setMouseCallback('Image', mouse_callback)
```

---

## Quick Tips

1. **Image Coordinates**: OpenCV uses `(x, y)` format where x is horizontal and y is vertical
2. **Color Format**: OpenCV uses BGR format by default, not RGB
3. **Array Indexing**: Access pixels with `img[y, x]` or `img[y, x, channel]`
4. **Data Type**: Images are NumPy arrays with dtype typically `uint8` (0-255)
5. **Waitkey Return**: `cv2.waitKey()` returns the ASCII value of key pressed
6. **Video Frame**: `ret` is boolean indicating if frame was read successfully
7. **Kernel Size**: Must be odd numbers (3, 5, 7, etc.) for most operations

---

## Common Imports
```python
import cv2
import numpy as np
```

**Note**: Always release resources using `cap.release()`, `out.release()`, and `cv2.destroyAllWindows()` to avoid memory leaks.
