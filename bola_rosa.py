import cv2
import numpy as np


def find_centroid(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return (cX, cY)
    else:
        return None


cap = cv2.VideoCapture(0)
ball_coords = None

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))

    if ret:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_pink = np.array([150, 50, 50])
        upper_pink = np.array([170, 255, 255])
        mask = cv2.inRange(hsv, lower_pink, upper_pink)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            centroid = find_centroid(max_contour)
            if centroid:
                ball_coords = centroid

        if ball_coords:
            ball_frame = np.zeros_like(frame)
            cv2.circle(ball_frame, ball_coords, 5, (0, 0, 255), -1)
            cv2.putText(
                ball_frame,
                f"({ball_coords[0]}, {ball_coords[1]})",
                (ball_coords[0] + 10, ball_coords[1] + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            rect_height = frame.shape[0] - 10
            rect_width = frame.shape[1] - 10
            cv2.rectangle(ball_frame, (5, 5), (rect_width, rect_height), (0, 255, 0), 2)
            cv2.imshow("Ball Position", ball_frame)

        cv2.imshow("Pink Ball Detection", frame)

        color_frame = np.zeros((100, 300, 3), dtype=np.uint8)
        color_frame[:, :] = cv2.cvtColor(
            np.uint8([[[160, 100, 100]]]), cv2.COLOR_HSV2BGR
        )
        cv2.imshow("Color Range", color_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
