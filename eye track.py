import cv2
import dlib
import numpy as np

# Load the pre-trained model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def midpoint(p1, p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def get_eye_points(shape, eye_indices):
    return [shape.part(i) for i in eye_indices]

def process_eye(eye_points):
    eye_region = np.array([(point.x, point.y) for point in eye_points], dtype=np.int32)
    return eye_region

def get_eye_center(eye_points):
    eye_region = process_eye(eye_points)
    M = cv2.moments(eye_region)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0
    return cx, cy

# Indices for left and right eyes in the shape predictor
left_eye_indices = [36, 37, 38, 39, 40, 41]
right_eye_indices = [42, 43, 44, 45, 46, 47]

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize game variables
ball_position = np.array([320, 240])
ball_radius = 20
speed = 5

def move_ball(direction, ball_position, speed):
    if direction == 'left':
        ball_position[0] -= speed
    elif direction == 'right':
        ball_position[0] += speed
    elif direction == 'up':
        ball_position[1] -= speed
    elif direction == 'down':
        ball_position[1] += speed
    return ball_position

def get_direction(left_eye_center, right_eye_center):
    dx = right_eye_center[0] - left_eye_center[0]
    dy = right_eye_center[1] - left_eye_center[1]
    if abs(dx) > abs(dy):
        return 'right' if dx > 0 else 'left'
    else:
        return 'down' if dy > 0 else 'up'

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)

        left_eye_points = get_eye_points(shape, left_eye_indices)
        right_eye_points = get_eye_points(shape, right_eye_indices)
        
        left_eye_center = get_eye_center(left_eye_points)
        right_eye_center = get_eye_center(right_eye_points)

        direction = get_direction(left_eye_center, right_eye_center)
        ball_position = move_ball(direction, ball_position, speed)

        cv2.circle(frame, (left_eye_center[0], left_eye_center[1]), 3, (0, 255, 0), -1)
        cv2.circle(frame, (right_eye_center[0], right_eye_center[1]), 3, (0, 255, 0), -1)

    # Ensure the ball stays within the window boundaries
    ball_position[0] = np.clip(ball_position[0], ball_radius, frame.shape[1] - ball_radius)
    ball_position[1] = np.clip(ball_position[1], ball_radius, frame.shape[0] - ball_radius)

    # Draw the ball
    cv2.circle(frame, tuple(ball_position), ball_radius, (255, 0, 0), -1)

    cv2.imshow('Eye Tracking Game', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
