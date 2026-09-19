#!/usr/bin/env python3
"""Real-time frontal face detection with OpenCV."""

import cv2


WINDOW_TITLE = "CodSoft Task 5 - Face Detection and Recognition"
CASCADE_FILE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def main():
    """Open the primary webcam and annotate detected faces until the user quits."""
    # Load the classifier shipped with OpenCV instead of relying on a local XML file.
    face_cascade = cv2.CascadeClassifier(CASCADE_FILE)
    if face_cascade.empty():
        raise RuntimeError(f"Could not load Haar cascade: {CASCADE_FILE}")

    # Camera index 0 is the system's primary webcam.
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            "Unable to open the primary webcam. Check that it is connected and available."
        )

    try:
        while True:
            success, frame = camera.read()
            if not success or frame is None:
                print("Unable to read a frame from the webcam.")
                break

            # Detection is faster on grayscale, while annotations remain on the color frame.
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray_frame,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
            )

            # Add a dark strip so the live face count stays readable on any background.
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 58), (20, 20, 20), -1)
            frame = cv2.addWeighted(overlay, 0.78, frame, 0.22, 0)
            cv2.putText(
                frame,
                f"Faces Detected: {len(faces)}",
                (18, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            for face_number, (x, y, width, height) in enumerate(faces, start=1):
                # LINE_AA produces smoother edges than the default rectangle rendering.
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + width, y + height),
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                label = f"Face {face_number} | {width}x{height}"
                label_y = max(y - 10, 78)
                cv2.putText(
                    frame,
                    label,
                    (x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(WINDOW_TITLE, frame)

            # Wait briefly for a key event; q is the documented clean-shutdown key.
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        # Always release hardware and GUI resources, including on an unexpected error.
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(f"Error: {error}")
