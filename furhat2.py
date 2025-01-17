from furhat_remote_api import FurhatRemoteAPI
import random
import time


class EmotionResponseSystem:
    def __init__(self, host="localhost", port="8080"):
        """Initialize the connection to the Furhat server."""
        self.url = f"http://{host}:{port}"
        print(f"Connecting to Furhat at {self.url}")
        
        try:
            self.furhat = FurhatRemoteAPI(self.url)
            # Verify the connection by trying an API call
            self.furhat.say(text="Testing connection to Furhat.")
            print("Successfully connected to Furhat.")
        except Exception as e:
            print(f"Connection error: {e}")
            raise

    def start_monitoring(self):
        """Start the emotion monitoring loop."""
        print("Starting emotion monitoring...")
        try:
            while True:
                try:
                    # Get the current users from Furhat
                    users = self.furhat.get_users()
                    
                    if users:
                        current_user = users[0]  # Get the first detected user
                        # Retrieve the user's emotion data
                        emotion = self.furhat.get_user_emotion(current_user.id)
                        
                        # Check emotions and respond accordingly
                        if emotion.sadness > 0.7:
                            self.handle_sadness()
                        elif emotion.anger > 0.7:
                            self.handle_anger()
                        elif emotion.joy > 0.7:
                            self.handle_happiness()
                        elif emotion.fear > 0.7:
                            self.handle_fear()
                        elif emotion.surprise > 0.7:
                            self.handle_surprise()
                    else:
                        print("No users detected. Waiting...")
                    
                except Exception as e:
                    print(f"Error during monitoring: {e}")
                
                time.sleep(1)  # Check every second
                
        except KeyboardInterrupt:
            print("\nStopping emotion monitoring...")
        finally:
            print("Shutting down the monitoring system.")

    def handle_sadness(self):
        """Handle sad emotion."""
        responses = [
            "I notice you seem a bit down. Would you like to talk about what's making you sad?",
            "Everyone has difficult times. Let me keep you company and chat.",
            "I'm here to listen if you want to share anything.",
            "Life can be challenging sometimes, but remember this is temporary.",
        ]
        self.furhat.say(text=random.choice(responses))
        self.furhat.gesture(name="Smile")

    def handle_anger(self):
        """Handle angry emotion."""
        responses = [
            "I understand you're angry. How about taking a deep breath to calm down?",
            "It's normal to feel angry, but don't let it cloud your judgment.",
            "Let's think about solutions together instead of dwelling on the anger.",
            "Can you tell me what's making you so angry? Maybe I can help.",
        ]
        self.furhat.say(text=random.choice(responses))
        self.furhat.gesture(name="Nod")

    def handle_happiness(self):
        """Handle happy emotion."""
        responses = [
            "Your happiness makes me happy too!",
            "That's a wonderful smile! Did something good happen?",
            "Keep that positive spirit going!",
            "It's great to see you so cheerful!",
        ]
        self.furhat.say(text=random.choice(responses))
        self.furhat.gesture(name="Smile")

    def handle_fear(self):
        """Handle fearful emotion."""
        responses = [
            "Don't be afraid, I'm here with you.",
            "Let's face what you're afraid of together.",
            "Can you tell me what's worrying you? Perhaps I can help.",
            "Remember, courage isn't the absence of fear, but overcoming it.",
        ]
        self.furhat.say(text=random.choice(responses))
        self.furhat.gesture(name="Nod")

    def handle_surprise(self):
        """Handle surprised emotion."""
        responses = [
            "Wow, you look surprised! What happened?",
            "That must have been unexpected! Want to share?",
            "Something seems to have caught you off guard!",
            "I'm curious about what surprised you so much!",
        ]
        self.furhat.say(text=random.choice(responses))
        self.furhat.gesture(name="Oh")


# Example usage
if __name__ == "__main__":
    # Create and start the emotion response system
    emotion_system = EmotionResponseSystem(host="localhost", port="8080")  # Replace with Furhat's actual IP/port if needed
    
    try:
        emotion_system.start_monitoring()
    except Exception as e:
        print(f"Fatal error: {e}")
