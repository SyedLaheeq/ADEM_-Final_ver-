import pickle
import cv2
import time
from collections import Counter
from utils import get_face_landmarks
from furhat_remote_api import FurhatRemoteAPI
import random

# Define emotion labels
emotions = ['ANGRY', 'FEAR', 'HAPPY', 'SAD']

# Load the pre-trained model
with open('./model', 'rb') as f:
    model = pickle.load(f)

# Start video capture
cap = cv2.VideoCapture(0)

# Initialize variables
start_time = time.time()
emotion_count = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get face landmarks
    face_landmarks = get_face_landmarks(frame, draw=True, static_image_mode=False)
    
    if face_landmarks:
        # Predict emotion
        output = model.predict([face_landmarks])
        predicted_emotion = emotions[int(output[0])]
        emotion_count.append(predicted_emotion)

        # Display predicted emotion
        cv2.putText(frame,
                    predicted_emotion,
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

    # Show frame
    cv2.imshow('frame', frame)
    
    # Check for 10-second duration
    if time.time() - start_time > 10:
        break

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()

# Find the most seen emotion
if emotion_count:
    most_common_emotion = Counter(emotion_count).most_common(1)[0][0]
    print(f"Most seen emotion: {most_common_emotion}")
else:
    print("No emotions detected.")

# Furhat Integration
class EmotionResponseSystem:
    def __init__(self, host="localhost"):
        # Connect to the Furhat robot using the correct URL format
        self.furhat = FurhatRemoteAPI("http://" + host + ":8080")

    def respond_to_emotion(self, emotion):
        responses = {
            'SAD': [
                "I notice you seem a bit down. Would you like to talk about what's making you sad?",
                "Everyone has difficult times. Let me keep you company and chat.",
                "I'm here to listen if you want to share anything.",
                "Life can be challenging sometimes, but remember this is temporary."
            ],
            'ANGRY': [
                "I understand you're angry. How about taking a deep breath to calm down?",
                "It's normal to feel angry, but don't let it cloud your judgment.",
                "Let's think about solutions together instead of dwelling on the anger.",
                "Can you tell me what's making you so angry? Maybe I can help."
            ],
            'HAPPY': [
                "Your happiness makes me happy too!",
                "That's a wonderful smile! Did something good happen?",
                "Keep that positive spirit going!",
                "It's great to see you so cheerful!"
            ],
            'FEAR': [
                "Don't be afraid, I'm here with you.",
                "Let's face what you're afraid of together.",
                "Can you tell me what's worrying you? Perhaps I can help.",
                "Remember, courage isn't the absence of fear, but overcoming it."
            ]
        }
        
        # Respond with appropriate Furhat dialogue
        if emotion in responses:
            self.furhat.say(text=random.choice(responses[emotion]))

# Start Furhat interaction for 20 seconds
emotion_system = EmotionResponseSystem("localhost")
furhat_start_time = time.time()

while time.time() - furhat_start_time < 20:
    if most_common_emotion:
        emotion_system.respond_to_emotion(most_common_emotion)
        break
