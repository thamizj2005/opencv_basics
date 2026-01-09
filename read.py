"""
OpenCV Basics - Practical Examples
Author: Thamizh
Purpose: Hands-on code for daily OpenCV practice

This file contains working examples of all fundamental OpenCV operations.
Uncomment the section you want to run and comment out the others.
"""

import cv2 as cv


# =============================================================================
# SECTION 1: READING AND DISPLAYING AN IMAGE
# =============================================================================
def read_image():
    """Read and display a static image"""
    print("📸 Reading image...")
    
    # Load image from file
    img = cv.imread('/home/thamizh/opencv_read/cars.png')
    
    if img is None:
        print("❌ Error: Could not read image. Check the path!")
        return
    
    # Display the image
    cv.imshow('Image Display', img)
    print("✅ Image loaded successfully. Press any key to close.")
    cv.waitKey(0)
    cv.destroyAllWindows()


# =============================================================================
# SECTION 2: WEBCAM CAPTURE
# =============================================================================
def read_webcam():
    """Capture and display live webcam feed"""
    print("📹 Starting webcam...")
    
    cap = cv.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not access webcam!")
        return
    
    print("✅ Webcam started. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break
        
        # Show frame
        cv.imshow('Webcam Feed', frame)
        
        # Exit on 'q' press
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()
    print("✅ Webcam stopped.")


# =============================================================================
# SECTION 3: VIDEO FILE PLAYBACK
# =============================================================================
def read_video():
    """Read and play a video file"""
    print("🎬 Loading video...")
    
    cap = cv.VideoCapture('/home/thamizh/opencv_read/1.mp4')
    
    if not cap.isOpened():
        print("❌ Error: Could not open video file!")
        return
    
    # Get FPS for proper playback speed
    fps = cap.get(cv.CAP_PROP_FPS)
    delay = int(1000 / fps)
    
    print(f"✅ Video loaded | FPS: {fps} | Delay: {delay}ms")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("✅ Video ended.")
            break
        
        cv.imshow('Video Playback', frame)
        
        if cv.waitKey(delay) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()


# =============================================================================
# SECTION 4: VIDEO WITH ANNOTATIONS (BOUNDING BOX + TEXT)
# =============================================================================
def video_with_annotations():
    """Play video with bounding box and text overlay"""
    print("🎬 Loading video with annotations...")
    
    cap = cv.VideoCapture('/home/thamizh/opencv_read/1.mp4')
    
    if not cap.isOpened():
        print("❌ Error: Could not open video file!")
        return
    
    fps = cap.get(cv.CAP_PROP_FPS)
    delay = int(1000 / fps)
    frame_count = 0
    
    print(f"✅ Video loaded | FPS: {fps}")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"✅ Video ended. Total frames: {frame_count}")
            break
        
        frame_count += 1
        
        # Resize for better performance
        frame = cv.resize(frame, (1024, 640))
        
        # Draw green bounding box
        cv.rectangle(frame, (100, 100), (350, 350), (0, 255, 0), 3)
        
        # Add text label
        cv.putText(
            frame,
            'Detection Zone',
            (110, 90),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # Add frame counter
        cv.putText(
            frame,
            f'Frame: {frame_count}',
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )
        
        cv.imshow('Annotated Video', frame)
        
        if cv.waitKey(delay) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()


# =============================================================================
# SECTION 5: COLOR vs GRAYSCALE COMPARISON
# =============================================================================
def color_vs_grayscale():
    """Display color and grayscale video side by side"""
    print("🎨 Starting color vs grayscale comparison...")
    
    cap = cv.VideoCapture('/home/thamizh/opencv_read/1.mp4')
    
    if not cap.isOpened():
        print("❌ Error: Could not open video file!")
        return
    
    fps = cap.get(cv.CAP_PROP_FPS)
    delay = int(1000 / fps)
    
    print("✅ Video loaded. Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize for performance
        frame = cv.resize(frame, (1024, 640))
        
        # Draw bounding box on color frame
        cv.rectangle(frame, (100, 100), (350, 350), (0, 255, 0), 3)
        cv.putText(
            frame,
            'Detection Zone',
            (500, 300),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )
        
        # Convert to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # Show both windows
        cv.imshow('COLOR', frame)
        cv.imshow('GRAY', gray)
        
        if cv.waitKey(delay) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()
    print("✅ Stopped.")


# =============================================================================
# SECTION 6: INTERACTIVE DRAWING DEMO
# =============================================================================
def interactive_drawing():
    """Draw multiple shapes and text on video"""
    print("🎨 Starting interactive drawing demo...")
    
    cap = cv.VideoCapture('/home/thamizh/opencv_read/1.mp4')
    
    if not cap.isOpened():
        print("❌ Error: Could not open video file!")
        return
    
    fps = cap.get(cv.CAP_PROP_FPS)
    delay = int(1000 / fps)
    
    print("✅ Drawing demo loaded. Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv.resize(frame, (1024, 640))
        
        # Draw multiple rectangles with different colors
        cv.rectangle(frame, (50, 50), (200, 200), (0, 255, 0), 3)      # Green
        cv.rectangle(frame, (250, 50), (400, 200), (255, 0, 0), 3)    # Blue
        cv.rectangle(frame, (450, 50), (600, 200), (0, 0, 255), 3)    # Red
        
        # Filled rectangle
        cv.rectangle(frame, (650, 50), (800, 200), (255, 255, 0), -1) # Cyan filled
        
        # Add labels
        cv.putText(frame, 'Green', (70, 230), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv.putText(frame, 'Blue', (270, 230), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv.putText(frame, 'Red', (480, 230), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv.putText(frame, 'Filled', (660, 230), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Title
        cv.putText(
            frame,
            'OpenCV Drawing Demo',
            (300, 450),
            cv.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            3
        )
        
        cv.imshow('Drawing Demo', frame)
        
        if cv.waitKey(delay) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()
    print("✅ Demo complete.")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("OpenCV Basics - Select an example to run")
    print("=" * 60)
    print("1. Read and display image")
    print("2. Webcam capture")
    print("3. Video playback")
    print("4. Video with annotations")
    print("5. Color vs Grayscale comparison")
    print("6. Interactive drawing demo")
    print("=" * 60)
    
    # CHANGE THIS NUMBER TO RUN DIFFERENT EXAMPLES
    choice = 5  # Currently set to run color vs grayscale
    
    if choice == 1:
        read_image()
    elif choice == 2:
        read_webcam()
    elif choice == 3:
        read_video()
    elif choice == 4:
        video_with_annotations()
    elif choice == 5:
        color_vs_grayscale()
    elif choice == 6:
        interactive_drawing()
    else:
        print("❌ Invalid choice! Please select 1-6")
    
    print("=" * 60)
    print("✅ Program finished.")
    print("=" * 60)