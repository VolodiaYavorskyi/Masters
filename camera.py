import cv2
import numpy as np
from keras.models import load_model

# Predefined class labels
emotion_labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# Load your model
model = load_model('cnn.h5')

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def preprocess_frame(face_frame):
    # Convert to grayscale
    gray_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2GRAY)
    # Resize to match FER2013 input size
    resized_frame = cv2.resize(gray_frame, (48, 48), interpolation=cv2.INTER_AREA)
    # Normalize pixel values
    normalized_frame = resized_frame / 255.0
    # Expand dimensions to match the model's input shape
    expanded_frame = np.expand_dims(normalized_frame, axis=0)
    expanded_frame = np.expand_dims(expanded_frame, axis=-1)
    return expanded_frame


# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Crop and preprocess the face
        face_frame = frame[y:y + h, x:x + w]
        processed_frame = preprocess_frame(face_frame)

        # Predict emotion
        prediction = model.predict(processed_frame)
        emotion_index = np.argmax(prediction)
        emotion = emotion_labels[emotion_index]

        # Overlay text on the frame
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('face', np.squeeze(processed_frame))

    # Display the original frame
    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
