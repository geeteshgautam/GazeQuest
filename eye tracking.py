import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load the data (assuming CSV format with columns 'time', 'x', 'y')
data = pd.read_csv('eye_tracking_data.csv')


# Calculate gaze velocity
data['vx'] = np.gradient(data['x'], data['time'])
data['vy'] = np.gradient(data['y'], data['time'])
data['velocity'] = np.sqrt(data['vx']2 + data['vy']2)


# Detect micro saccades based on a velocity threshold
velocity_threshold = 50  # This value may need to be adjusted
micro_saccades = data[data['velocity']  velocity_threshold]


# Plot gaze position and detected micro saccades
plt.figure(figsize=(12, 6))
plt.plot(data['time'], data['x'], label='Gaze X Position')
plt.plot(data['time'], data['y'], label='Gaze Y Position')
plt.scatter(micro_saccades['time'], micro_saccades['x'], color='r', label='Micro Saccades X')
plt.scatter(micro_saccades['time'], micro_saccades['y'], color='g', label='Micro Saccades Y')
plt.xlabel('Time (s)')
plt.ylabel('Gaze Position')
plt.legend()
plt.title('Gaze Position and Micro Saccades')
plt.show()


# Analyze micro saccade metrics
micro_saccades['duration'] = micro_saccades['time'].diff().fillna(0)
micro_saccades_summary = micro_saccades.describe()


print(micro_saccades_summary)


import cv2
import numpy as np


# Initialize the IR camera
camera = cv2.VideoCapture(0)  # Change index as per your camera setup


# Set camera properties (adjust frame rate and resolution as needed)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_FPS, 30)


while True:
    ret, frame = camera.read()
    if not ret:
        break
    
    # Convert frame to grayscale for easier processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to highlight gas bubbles
    _, thresholded = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    
    # Find contours (blobs) in the thresholded image
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on the original frame
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Display the processed frame
    cv2.imshow('IR Eye Tracking', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release the camera and close windows
camera.release()
cv2.destroyAllWindows()


#c# code
using Windows.Devices.Input.Preview;


public async Task StartGazeTrackingAsync()
{
    var gazeInputSourcePreview = await GazeInputSourcePreview.GetForCurrentViewAsync();
    gazeInputSourcePreview.GazeEntered += GazeInputSourcePreview_GazeEntered;
    gazeInputSourcePreview.GazeMoved += GazeInputSourcePreview_GazeMoved;
    gazeInputSourcePreview.GazeExited += GazeInputSourcePreview_GazeExited;
}


private void GazeInputSourcePreview_GazeEntered(
    GazeInputSourcePreview sender,
    GazeEnteredPreviewEventArgs args)
{
    var gazePoint = args.CurrentPoint.EyeGazePosition;
    // Process gazePoint to get initial position
}


private void GazeInputSourcePreview_GazeMoved(
    GazeInputSourcePreview sender,
    GazeMovedPreviewEventArgs args)
{
    var gazePoint = args.CurrentPoint.EyeGazePosition;
    // Process gazePoint to update position and calculate velocity
}


private void GazeInputSourcePreview_GazeExited(
    GazeInputSourcePreview sender,
    GazeExitedPreviewEventArgs args)
{
    // Handle gaze exit
}