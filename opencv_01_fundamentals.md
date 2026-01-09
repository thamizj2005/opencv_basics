# OpenCV Daily Revision Guide (Python)
## Part 1: Fundamentals

**Author:** Thamizh  
**Purpose:** Daily revision and long-term reference  
**Level:** Absolute Beginner to Strong Foundation  

This guide covers every OpenCV concept I've learned so far, with complete code examples, syntax, and clear explanations. Nothing is assumed and nothing is skipped.

---

## What is OpenCV?

OpenCV (Open Source Computer Vision) is a powerful library for:

- Reading images
- Reading videos
- Accessing the webcam
- Processing frames
- Drawing shapes
- Preparing data for AI models (YOLO, CNN, etc.)

In Python, we use it through the `cv2` module.

---

## Installing OpenCV

```bash
pip install opencv-python
```

This installs the main OpenCV package for Python.

---

## Importing OpenCV

```python
import cv2 as cv
```

- `cv2` is the module name.
- `as cv` is a common shortcut (almost everyone uses it).
- All OpenCV functions will be called as `cv.function_name()`.

---

## Reading an Image

```python
img = cv.imread('/path/to/your/image.jpg')
```

- Loads the image from disk.
- Returns a NumPy array.
- **Important:** Images are loaded in **BGR order** (Blue-Green-Red), not RGB.

---

## Displaying the Image

```python
cv.imshow('Image', img)
cv.waitKey(0)
cv.destroyAllWindows()
```

- `imshow()` opens a window with the given title.
- `waitKey(0)` waits indefinitely until any key is pressed.
- `destroyAllWindows()` closes all open windows properly.

---

## Reading Video or Webcam

### Webcam

```python
cap = cv.VideoCapture(0)
```

- `0` = default laptop webcam
- `1, 2, ...` = external cameras

### Video File

```python
cap = cv.VideoCapture('/path/to/your/video.mp4')
```

Use absolute paths to avoid issues.

---

## Understanding `cap.read()`

```python
ret, frame = cap.read()
```

- `ret`: Boolean – `True` if frame was read successfully, `False` if end of video or error.
- `frame`: The actual image (NumPy array).

---

## Standard Video Reading Loop

```python
while True:
    ret, frame = cap.read()
    if not ret:
        break
```

This loop stops automatically when the video ends and prevents errors on empty frames.

---

## Showing Video Frames

```python
cv.imshow('Frames', frame)
```

Placed inside the loop, it updates the window rapidly to create the video playback effect.

---

## Keyboard Control and Exit

```python
if cv.waitKey(1) & 0xFF == ord('q'):
    break
```

- `waitKey(1)` waits 1 ms for a key press.
- Pressing `'q'` breaks the loop and exits.

---

## Releasing Resources (Very Important!)

```python
cap.release()
cv.destroyAllWindows()
```

Always do this at the end:

- Frees the camera/video file.
- Prevents memory leaks and port locks.

---

## FPS (Frames Per Second)

```python
fps = cap.get(cv.CAP_PROP_FPS)
```

Gets the original FPS from video metadata – needed for correct playback speed.

---

## Correct Video Speed Control

```python
delay = int(1000 / fps)
```

- `waitKey()` uses milliseconds.
- 1000 ms = 1 second.
- Example: FPS = 25 → delay = 40 ms per frame.

Use `cv.waitKey(delay)` instead of a fixed value.

---

## Resize Frame (For Better Performance)

```python
frame = cv.resize(frame, (1024, 640))
```

- Format: `(width, height)`
- Smaller resolution = faster processing
- Larger = more detail but slower

---

## OpenCV Coordinate System

- **Origin (0, 0):** Top-left corner
- **X-axis:** left → right
- **Y-axis:** top → bottom
- Points are given as `(x, y)`

---

## Drawing a Rectangle (Bounding Box)

### Standard Border

```python
cv.rectangle(frame, (100, 100), (350, 350), (0, 255, 0), 3)
```

- `(100, 100)`: top-left corner
- `(350, 350)`: bottom-right corner
- `(0, 255, 0)`: Green color (BGR)
- `3`: border thickness in pixels

### Filled Rectangle

```python
cv.rectangle(frame, (100, 100), (350, 350), (0, 255, 0), -1)
```

Use `-1` for thickness to fill the rectangle.

---

## Adding Text on Frame

```python
cv.putText(
    frame,
    'Simple box',
    (500, 300),
    cv.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 0, 255),
    2
)
```

- Position `(500, 300)`: bottom-left corner of text
- Font scale `1`: size
- `(0, 0, 255)`: Red color (BGR)
- Thickness `2`

---

## Common Fonts

The most used one is `cv.FONT_HERSHEY_SIMPLEX`. Others exist, but this is clean and reliable.

---

## Color Conversion (BGR to Gray)

```python
gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
```

- Converts 3-channel BGR to 1-channel grayscale.
- Much faster for many algorithms (edges, contours, etc.).
- Required preprocessing for most ML models.

---

## Showing Multiple Windows

```python
cv.imshow('COLOR', frame)
cv.imshow('GRAY', gray)
```

You can open as many windows as needed – just give each a unique title.

---

## Complete Working Program

```python
import cv2 as cv

# Open video (change path to your own video)
cap = cv.VideoCapture('/path/to/your/video.mp4')

# Get FPS and calculate delay
fps = cap.get(cv.CAP_PROP_FPS)
delay = int(1000 / fps)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for performance
    frame = cv.resize(frame, (1024, 640))

    # Draw rectangle
    cv.rectangle(frame, (100, 100), (350, 350), (0, 255, 0), 3)

    # Add text
    cv.putText(frame, 'Simple box', (500, 300),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Convert to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Show both windows
    cv.imshow('COLOR', frame)
    cv.imshow('GRAY', gray)

    # Quit on 'q'
    if cv.waitKey(delay) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv.destroyAllWindows()
```

---

## Key Rules to Remember

1. OpenCV uses **BGR**, not RGB.
2. Coordinates are `(x, y)`, not `(row, column)`.
3. Always check `ret` after `cap.read()`.
4. Always call `cap.release()` at the end.
5. Use proper FPS-based delay for natural playback speed.
6. Rectangle + text = core of object detection visualization.

---

## What You've Learned So Far

- ✅ Reading images
- ✅ Reading video & webcam
- ✅ Handling FPS and playback speed
- ✅ Resizing frames
- ✅ Coordinate system
- ✅ Drawing rectangles
- ✅ Adding text
- ✅ Grayscale conversion

---

## Next Steps

Continue to **Part 2: Advanced Drawing & Detection** for:

- Drawing lines
- Line crossing logic
- Object counting
- YOLO integration
- Object tracking

---

**Last Updated:** 2026-01-09  
**Status:** Complete - Ready for Daily Revision
