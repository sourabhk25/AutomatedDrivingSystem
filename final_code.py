import cv2
import numpy as np
import time
import imutils
import threading

# Suppress divide by zero warnings
np.seterr(divide='ignore', invalid='ignore')

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def draw_lines(img, lines, color=(0, 255, 0), thickness=3):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

def extrapolate_line(line, height):
    x1, y1, x2, y2 = line[0]
    slope = (y2 - y1) / (x2 - x1)
    y1 = int(0.8 * height) 
    x1 = int(x2 - (y2 - y1) / slope)
    y2 = int(0.8 * height) + 500
    x2 = int(x1 + (y2 - y1) / slope)
    return np.array([[x1, y1, x2, y2]], dtype=np.int32)

def lane_detection(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Define the region of interest
    height, width = edges.shape
    roi_vertices = [
        (0, height),
        (width // 2, height // 2),
        (width, height)
    ]
    roi_edges = region_of_interest(edges, np.array([roi_vertices], np.int32))

    # Apply Hough Transform to detect lines
    lines = cv2.HoughLinesP(
        roi_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=20,
        maxLineGap=100
    )

    # Separate left and right lane lines based on slope
    left_lines = []
    right_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if slope < -0.5:
                left_lines.append(line)
            elif slope > 0.5:
                right_lines.append(line)

    # Create a mask for the road area between the lane lines
    road_mask = np.zeros_like(frame)

    # Check if left and right lane lines are detectedßßß
    if len(left_lines) > 0 and len(right_lines) > 0:
        # Average and extrapolate the left and right lane lines
        left_lane = np.average(left_lines, axis=0)
        right_lane = np.average(right_lines, axis=0)

        # Extrapolate left and right lines
        height, _ = frame.shape[:2]  # Get the height of the frame
        y_bottom = height  # Set the bottom y-coordinate to the frame height
        left_lane = extrapolate_line(left_lane, y_bottom)
        right_lane = extrapolate_line(right_lane, y_bottom)

        # Create a blank image with the same dimensions as the original image
        line_img = np.zeros_like(frame)

        # Draw the left and right lane lines
        draw_lines(line_img, [left_lane], color=(0, 0, 255), thickness=10)
        draw_lines(line_img, [right_lane], color=(0, 0, 255), thickness=10)

        road_polygon = np.array([
            (left_lane[0][0], left_lane[0][1]),
            (right_lane[0][0], right_lane[0][1]),
            (right_lane[0][2], right_lane[0][3]),
            (left_lane[0][2], left_lane[0][3])
        ])
        cv2.fillPoly(road_mask, np.int32([road_polygon]), (0, 255, 0))
    
    else:
        # If either left or right lane line is missing, skip the extrapolation and drawing steps
        line_img = np.zeros_like(frame)

     # Combine the original frame with the line image and road mask
    annotated_frame = cv2.addWeighted(frame, 0.8, line_img, 1, 0)
    annotated_frame = cv2.addWeighted(annotated_frame, 1, road_mask, 0.3, 0)

    return annotated_frame

def pedestrian_detection(frame):
    body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')

    # Resize the frame 
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

    # Detect pedestrians
    pedestrians = body_classifier.detectMultiScale(frame, 1.2, 3)

    # Draw bounding boxes around pedestrians
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    return frame 


def stop_sign_detection(frame):
    stop_sign_cascade = cv2.CascadeClassifier('cascade_stop_sign.xml')

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect stop signs
    # stop_sign_scaled = stop_sign.detectMultiScale
    stop_signs = stop_sign_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw bounding boxes around stop signs
    for (x, y, w, h) in stop_signs:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(frame, "Stop Sign", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame

def process_frames(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Read the first frame
    ret, frame = cap.read()
    frame_count = 0
    start_time = time.time()

    while ret:
        # Process the frame
        processed_frame = frame.copy()
        processed_frame = lane_detection(processed_frame)
        processed_frame = stop_sign_detection(processed_frame)
        processed_frame = pedestrian_detection(processed_frame)
        
        # Display the processed frame
        cv2.imshow("Processed Frame", processed_frame)

        # Read the next frame
        ret, frame = cap.read()
        frame_count += 1

        # Calculate the current frame rate
        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time > 0:
            current_fps = frame_count / elapsed_time
            # print("Current FPS: {:.2f}".format(current_fps))
        cv2.putText(frame, f"FPS: {current_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Check for the 'q' key to stop the processing
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Calculate the average frame rate
    end_time = time.time()
    duration = end_time - start_time
    average_fps = frame_count / duration
    print("Average FPS: {:.2f}".format(average_fps))

    # Release the video file and close windows
    cap.release()
    cv2.destroyAllWindows()


# Main thread
if __name__ == '__main__':
    video_path = 'lane-stop-pedestrian.mp4'
    
    process_frames(video_path)

