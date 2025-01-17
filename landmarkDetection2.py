import cv2
import mediapipe as mp

# Initialize holistic and face_mesh models
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Define a function to detect landmarks
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False  # Set image as non-writable to speed up the inference
    results = model.process(image)
    image.flags.writeable = True  # Set image as writable again
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Function to draw landmarks with connections
def draw_landmarks(image, results):
    # Draw face landmarks with FACEMESH_TESSELATION
    if results.face_landmarks:
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)
    
    # Draw landmarks for pose, hands, etc.
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

# Main function to capture video and process frames
cap = cv2.VideoCapture(0)

# Initialize Holistic model and FaceMesh model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic, \
     mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        ret, frame = cap.read()

        # Make detections for holistic landmarks
        image, results = mediapipe_detection(frame, holistic)

        # Draw landmarks and connections
        draw_landmarks(image, results)

        # Display the frame with landmarks drawn
        cv2.imshow('MediaPipe Holistic and FaceMesh', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
