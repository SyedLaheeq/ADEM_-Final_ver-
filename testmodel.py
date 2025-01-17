import pickle
import cv2
import time
from collections import Counter
from utils import get_face_landmarks
from furhat_remote_api import FurhatRemoteAPI
import random
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer  # TensorFlow-compatible GPT-2

# Initialize the TensorFlow-based GPT-2 model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt_model = TFGPT2LMHeadModel.from_pretrained("gpt2")  # GPT-2 model

# Function to get LLM-generated response for the "HAPPY" emotion
def get_llm_response(emotion):
    """
    Generate a response using GPT-2 (TensorFlow) for the given emotion.
    """
    try:
        # Define a better prompt
        prompt = f"""
        The user is feeling {emotion}.
        As a friendly and caring companion, provide a response to make them feel supported and engaged.
        """.strip()

        # Check and add `pad_token` if missing
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            gpt_model.resize_token_embeddings(len(tokenizer))  # Update model token embeddings

        # Tokenize the input with padding and truncation
        inputs = tokenizer(prompt, return_tensors="tf", padding=True, truncation=True)

        # Generate a response with improved settings
        outputs = gpt_model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=50,
            do_sample=True,
            temperature=0.9,  # Encourage creativity
            top_k=50,  # Filter unlikely tokens
            top_p=0.95,  # Nucleus sampling
            pad_token_id=tokenizer.pad_token_id
        )

        # Decode the generated response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        print("Response:", response)

        # Post-process the response to remove the original prompt if echoed
        if prompt in response:
            response = response.replace(prompt, "").strip()

        return response if response else "I'm here to share your happiness!"

    except Exception as e:
        print(f"Error with LLM: {e}")
        return "Your happiness makes me happy too!"

# Create an instance of the FurhatRemoteAPI class
furhat = FurhatRemoteAPI("localhost")

# Get the voices on the robot
voices = furhat.get_voices()

# Get the users on the robot
users = furhat.get_users()

# Get the current gestures of the robot
gestures = furhat.get_gestures()

# Predefined responses for emotions other than "HAPPY"

# Set voice
furhat.set_face(character="Lamin", mask="adult")
furhat.set_voice(name="Brian")

# Define emotion labels
emotions = ['ANGRY', 'FEAR', 'HAPPY', 'SAD']

# Load the pre-trained emotion detection model (RandomForestClassifier) as emotion_model
with open('./model', 'rb') as f:
    emotion_model = pickle.load(f)  # RandomForest model for emotion detection

# Start video capture
cap = cv2.VideoCapture(0)

# Function to detect emotions for a given duration
def detect_emotions(duration):
    start_time = time.time()
    emotion_count = []
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break

        # Get face landmarks
        face_landmarks = get_face_landmarks(frame, draw=True, static_image_mode=False)
        
        if face_landmarks:
            # Predict emotion using the RandomForest model
            output = emotion_model.predict([face_landmarks])  # Using emotion_model for emotion detection
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

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return None

    return Counter(emotion_count).most_common(1)[0][0] if emotion_count else None

# Main loop to alternate between detection and response
try:
    while True:
        # Run emotion detection for 10 seconds
        emotion = detect_emotions(10)

        # Respond to detected emotion for 10 seconds
        if emotion:
            print(f"Most seen emotion: {emotion}")
            if emotion == "HAPPY":
                response = get_llm_response(emotion)  # Use GPT-2 only for "HAPPY"
                furhat.say(text=response, blocking=True)
            elif emotion == "SAD":
            # Predefined comforting responses
                furhat.say(text="Why are feeling Sad? you can share it with me", blocking=True)
                furhat.gesture(name="Nod")
                time.sleep(5)
                furhat.say(text="Its ok not to be Ok. it will only get better. Do you need any more help")
                try:
                    result = furhat.listen() 
                    time.sleep(5)
                    furhat.listen_stop()
                    if hasattr(result, "message") and result.message:
                        print(f"User said: {result.message}")
                        if "yes" in result.message.lower():
                            furhat.say(text="Life can be challenging sometimes, but remember this is temporary.")
                            time.sleep(1)
                        else:
                            furhat.say(text="Ok. No problem! Let me know if there's something else you'd like.")
                except Exception as e:
                    print(f"Error: {e}")
                    furhat.say(text="There was an issue understanding you. Please try again.")
            elif emotion == "ANGRY":
            # Predefined calming responses
                furhat.say(text="Why are feeling Angry? you can share it with me", blocking=True)
                furhat.gesture(name="Nod")
                time.sleep(5)
                furhat.say(text="Anger is bad cause. it will only make things worse. Do you need any more help")
                try:
                    result = furhat.listen() 
                    time.sleep(5)
                    furhat.listen_stop()
                    if hasattr(result, "message") and result.message:
                        print(f"User said: {result.message}")
                        if "yes" in result.message.lower():
                            furhat.say(text="It's normal to feel angry, but don't let it cloud your judgment. Chill out and Roll out")
                            time.sleep(1)
                        else:
                            furhat.say(text="ok. No problem! Let me know if there's something else you'd like.")
                            furhat.gesture(name="ShakeHead")
                except Exception as e:
                    print(f"Error: {e}")
                    furhat.say(text="There was an issue understanding you. Please try again.")
            elif emotion == "FEAR":
            # Predefined encouraging responses
                furhat.say(text="Why are feeling scared? you can share it with me", blocking=True)
                furhat.gesture(name="Nod")
                time.sleep(3)
                furhat.say(text="Remember, courage isn't the absence of fear, but overcoming it. Do you need any more help")
                try:
                    result = furhat.listen() 
                    time.sleep(3)
                    furhat.listen_stop()
                    if hasattr(result, "message") and result.message:
                        print(f"User said: {result.message}")
                        if "yes" in result.message.lower():
                            furhat.say(text="Ability to overcome fear is what makes you brave and you are very brave. Dont forget")
                            time.sleep(1)
                        else:
                            furhat.say(text="ok. No problem! Let me know if there's something else you'd like.")
                except Exception as e:
                    print(f"Error: {e}")
                    furhat.say(text="There was an issue understanding you. Please try again.")
            else:
            # Default response for unrecognized emotions
                response = "I’m here to help, no matter how you feel. Can you tell me more?"
                furhat.say(text=response)
            time.sleep(23)  # Response duration
        else:
            print("No emotions detected.")

except KeyboardInterrupt:
    print("Exiting the program...")
    furhat.say(text="Do you feel better now after talking to me?")
    time.sleep(3)
    try:
        result = furhat.listen() 
        furhat.listen_stop()
        if hasattr(result, "message") and result.message:
            print(f"User said: {result.message}")
            if "yes" in result.message.lower():
                furhat.say(text="Hope I was of help to u. Have a great da Ahead!")
                furhat.gesture(name="Nod")
                furhat.listen_stop()
            else:
                furhat.say(text="Sorry I couldn't be much of any help to you. I will be better next time, you can give a suggestion.")
                time.sleep(3)
                result = furhat.listen() 
                furhat.listen_stop()
                print("Suggestion by the user is: ", result.message)
    except Exception as e:
        print(f"Error: {e}")
        furhat.say(text="There was an issue understanding you. Please try again.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
