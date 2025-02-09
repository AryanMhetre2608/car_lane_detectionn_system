import cv2
import numpy as np

# Path to your video file
video_src = 'video_car_lane.mp4'

# Variables to store previous frame's lane for smoothing
prev_left_line = None #reduce jitter between frame 
prev_right_line = None

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(img, mask)

# This function draws lines on the image (lanes in this case).
def draw_lines(img, lines, color=(0, 255, 0), thickness=2):
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    return cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)

# This function calculates the slope of a line segment defined by two points
def get_slope(x1, y1, x2, y2):
    return (y2 - y1) / (x2 - x1 + np.finfo(float).eps)

# This is the core function that processes each video frame to detect the lanes.
def process_lanes(image):
    global prev_left_line, prev_right_line  # Use global variables for lane history

    height, width = image.shape[:2]
    roi_vertices = np.array([[(0, height), (width * 0.45, height * 0.6), 
                              (width * 0.55, height * 0.6), (width, height)]], dtype=np.int32)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Adaptive thresholding to handle different lighting conditions
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Edge detection
    blur = cv2.GaussianBlur(thresh, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    roi = region_of_interest(edges, roi_vertices)
    
    lines = cv2.HoughLinesP(roi, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=30)
    
    if lines is None:
        return image
    
    left_lines, right_lines = [], []
    # Lane-segmentation
    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = get_slope(x1, y1, x2, y2)
        if abs(slope) < 0.2:  # Filter out nearly horizontal lines (increased slope threshold)
            continue
        if slope < 0:
            left_lines.append(line)
        else:
            right_lines.append(line)
    
    def average_lines(lines):
        avg_line = np.zeros((4,))
        if len(lines) > 0:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    avg_line += [x1, y1, x2, y2]
            avg_line = avg_line / len(lines)
        return avg_line
    
    left_avg = average_lines(left_lines)
    right_avg = average_lines(right_lines)
    
    def extend_line(line, y_min, y_max):
        if np.all(line == 0):
            return None
        x1, y1, x2, y2 = map(int, line)
        slope = get_slope(x1, y1, x2, y2)
        if abs(slope) < 0.1:  # Avoid division by very small numbers
            return None
        x_min = int((y_min - y1) / slope + x1)
        x_max = int((y_max - y1) / slope + x1)
        return [x_min, y_min, x_max, y_max]
    
    y_min = int(height * 0.6)
    y_max = height
    left_line = extend_line(left_avg, y_min, y_max)
    right_line = extend_line(right_avg, y_min, y_max)
    
    # Smoothing to reduce jitter in right lane detection
    alpha = 0.8  # Smoothing factor, can adjust between 0 and 1
    
    if left_line and prev_left_line is not None:
        left_line = [
            int(alpha * prev_left_line[0] + (1 - alpha) * left_line[0]),
            int(alpha * prev_left_line[1] + (1 - alpha) * left_line[1]),
            int(alpha * prev_left_line[2] + (1 - alpha) * left_line[2]),
            int(alpha * prev_left_line[3] + (1 - alpha) * left_line[3])
        ]
    
    if right_line and prev_right_line is not None:
        right_line = [
            int(alpha * prev_right_line[0] + (1 - alpha) * right_line[0]),
            int(alpha * prev_right_line[1] + (1 - alpha) * right_line[1]),
            int(alpha * prev_right_line[2] + (1 - alpha) * right_line[2]),
            int(alpha * prev_right_line[3] + (1 - alpha) * right_line[3])
        ]
    
    # Update previous lanes for the next frame
    if left_line:
        prev_left_line = left_line
    if right_line:
        prev_right_line = right_line
    
    lanes = []
    if left_line:
        lanes.append([left_line])
    if right_line:
        lanes.append([right_line])
    
    return draw_lines(image, lanes, thickness=5)

# Capture video from file
cap = cv2.VideoCapture(video_src)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame to detect lanes
    lane_frame = process_lanes(frame)

    # Display the processed frame
    cv2.imshow('Robust High Accuracy Lane Detection', lane_frame)

    # Press 'q' to exit the loop and close the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()