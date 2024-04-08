import cv2
import face_recognition
import os
import pickle

def capture_face_image():
    # Start the webcam
    cap = cv2.VideoCapture(0)
    print("Press SPACE to capture the photo, or ESC to exit.")

    while True:
        ret, frame = cap.read()
        cv2.imshow('Capture', frame)

        key = cv2.waitKey(1)
        if key & 0xFF == 27:  # ESC key
            return None
        elif key & 0xFF == 32:  # SPACE key
            cap.release()
            cv2.destroyAllWindows()
            return frame

def register_face():
    user_id = input("Enter a unique ID for the user: ")
    frame = capture_face_image()

    if frame is not None:
        # Create a directory for the user if it doesn't exist
        user_dir = f'user_data/{user_id}'
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)

        image_path = os.path.join(user_dir, f'{user_id}.jpg')
        
        # Save the captured image
        cv2.imwrite(image_path, frame)

        # Load the user's image
        user_image = face_recognition.load_image_file(image_path)

        # Encode the user's face. Assuming each image has one face.
        face_encodings = face_recognition.face_encodings(user_image)
        if face_encodings:
            user_face_encoding = face_encodings[0]

            # Save the face encoding and user ID
            with open(os.path.join(user_dir, f'{user_id}_data.dat'), 'wb') as user_data_file:
                pickle.dump(user_face_encoding, user_data_file)

            print(f'User {user_id} registered successfully.')
        else:
            print("No faces detected in the image. Try again.")
            # Optionally, remove the created directory if registration fails
            # os.rmdir(user_dir)
    else:
        print("Registration cancelled.")

# Example usage:
register_face()
