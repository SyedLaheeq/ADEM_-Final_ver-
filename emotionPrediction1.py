import cv2
import mediapipe as mp
import time
import numpy as np
from mediapipe.python.solutions.face_mesh import FACEMESH_TESSELATION  # Import FACE_CONNECTIONS

# ----------------------------- Emotion Recognition Model (Dummy Example) -----------------------------
# You should replace this with a pre-trained emotion recognition model
# For example, a CNN trained on facial landmarks or pixel data
def predict_emotion(landmarks):
    # Dummy prediction function: Replace this with your model's prediction logic
    # Example: Use the model to predict the emotion based on the landmarks
    # Here, we just return a random emotion for the sake of example
    emotions = ['Happy', 'Sad', 'Angry', 'Surprised', 'Neutral']
    return emotions[np.random.randint(0, len(emotions))]

# ----------------------------- FaceLandMarks Class -----------------------------
class FaceLandMarks:
    def __init__(self, staticMode=False, maxFace=2, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFace = maxFace
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.staticMode,
            max_num_faces=self.maxFace,
            min_detection_confidence=self.minDetectionCon,
            min_tracking_confidence=self.minTrackCon
        )
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)
    
    def findFaceLandmark(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(imgRGB)

        faces = []

        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, FACEMESH_TESSELATION, self.drawSpec, self.drawSpec)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append([id, x, y])
                faces.append(face)

        return img, faces

# ----------------------------- Real-Time Face Landmark Detection -----------------------------
def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceLandMarks()

    while True:
        success, img = cap.read()
        if not success:
            print("Error: Could not read frame from camera.")
            break

        img, faces = detector.findFaceLandmark(img, draw=True)

        # Print the number of faces detected
        if len(faces) != 0:
            print(f"Number of faces detected: {len(faces)}")

            for face in faces:
                print(face)  # Print the landmarks for each face (ID, X, Y)

                # Preprocess the landmarks for emotion recognition
                landmarks = np.array([lm[1:] for lm in face])  # Extract (x, y) values
                landmarks = landmarks.flatten()  # Flatten into a 1D array

                # Predict the emotion from the landmarks
                emotion = predict_emotion(landmarks)
                print(f"Predicted Emotion: {emotion}")

                # Display the predicted emotion on the image
                cv2.putText(img, emotion, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Calculate and display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the result
        cv2.imshow("Face Landmark Detection", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Terminating the program...")
            break

    cap.release()
    cv2.destroyAllWindows()

# ----------------------------- Run the Program -----------------------------
if __name__ == "__main__":
    main()
