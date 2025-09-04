import cv2

# Open the first webcam (index 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Webcam not accessible")
else:
    print("✅ Webcam opened successfully. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame")
            break

        cv2.imshow("Webcam Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
