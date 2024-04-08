import cv2
import face_recognition
import os
import pickle

def capture_face_image():
    cap = cv2.VideoCapture(0)
    print("Press SPACE to capture the photo, or ESC to exit.")

    while True:
        ret, frame = cap.read()
        cv2.imshow('Capture', frame)

        key = cv2.waitKey(1)
        if key & 0xFF == 27:  # ESC key
            cap.release()
            cv2.destroyAllWindows()
            return None
        elif key & 0xFF == 32:  # SPACE key
            cap.release()
            cv2.destroyAllWindows()
            return frame

def recognize_face():
    frame = capture_face_image()
    if frame is not None:
        # Convert the captured frame to RGB as face_recognition expects
        rgb_frame = frame[:, :, ::-1]

        # Attempt to find faces in the captured frame
        face_locations = face_recognition.face_locations(rgb_frame)
        if not face_locations:
            print("No faces detected in the image.")
            return

        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Iterate through all registered users
        for user_id in os.listdir('user_data'):
            user_dir = os.path.join('user_data', user_id)
            user_data_file = os.path.join(user_dir, f'{user_id}_data.dat')

            if os.path.exists(user_data_file):
                with open(user_data_file, 'rb') as user_data:
                    registered_face_encoding = pickle.load(user_data)

                    # Compare the face encoding with the registered face
                    matches = face_recognition.compare_faces([registered_face_encoding], face_encodings[0])
                    if True in matches:
                        print(f'User {user_id} recognized.')
                        open_folder(user_dir)
                        return
        print('User not recognized.')
    else:
        print("Identification cancelled.")

def open_folder(folder_path):
    # Adjust the command below if not on Windows
    os.startfile(folder_path)
    print(f"Opened {folder_path}")

# Example usage:
recognize_face()
