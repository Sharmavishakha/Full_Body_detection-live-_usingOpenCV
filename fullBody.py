import cv2
# Load the pre-trained Haar Cascade classifier for full body detection
detector = cv2.CascadeClassifier('haarcascades/haarcascade_fullbody.xml')

# Open the default camera (usually the webcam)
cap = cv2.VideoCapture(0)

# Define a unique ID for saving captured images
Id = "out_sample"
while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Perform body detection on the frame
    bodies = detector.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in bodies:
        # Draw a rectangle around the detected body
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Save the captured body into a directory
        cv2.imwrite(r"C:\Users\chetna\Documents\captured_image\{}_{}.jpg".format(Id, len(bodies)), frame[y:y+h, x:x+w])
        
    # Show the frame with the rectangle surrounding the body
    cv2.imshow('frame', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()