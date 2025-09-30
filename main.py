import cv2
import numpy as np

def order_points(pts):
    """
    Initializes a list of coordinates that will be ordered
    such that the first entry in the list is the top-left,
    the second entry is the top-right, the third is the
    bottom-right, and the fourth is the bottom-left.
    """
    rect = np.zeros((4, 2), dtype="float32")

    # The top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum.
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference.
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # Return the ordered coordinates.
    return rect

def four_point_transform(image, pts):
    """
    Applies a four-point perspective transform to obtain a
    top-down view of an image.
    """
    # Obtain a consistent order of the points and unpack them individually.
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # or top-right and top-left x-coordinates.
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates.
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left order.
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Compute the perspective transform matrix and then apply it.
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # Return the warped image.
    return warped

def main():
    """
    Main function to run the card scanner.
    """
    # Initialize the video capture.
    # Use 0 for the default laptop camera.
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    while True:
        # Capture frame-by-frame.
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        # Make a copy of the original frame for displaying the final result.
        original_frame = frame.copy()

        # Convert the frame to grayscale.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise.
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Use Canny edge detection.
        edged = cv2.Canny(blurred, 75, 200)

        # Find contours in the edged image.
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours by area and keep the largest ones.
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        screenCnt = None

        # Loop over the contours.
        for c in contours:
            # Approximate the contour.
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            # If our approximated contour has four points, then we
            # can assume that we have found our card.
            if len(approx) == 4:
                screenCnt = approx
                break

        # If a four-point contour is found, draw it.
        if screenCnt is not None:
            cv2.drawContours(frame, [screenCnt], -1, (0, 255, 0), 2)

            # Apply the four point transform to get a top-down view.
            warped = four_point_transform(original_frame, screenCnt.reshape(4, 2))

            # Show the warped image.
            cv2.imshow("Warped", cv2.resize(warped, (300, 450))) # Resized for better viewing

        # Display the resulting frame with contours.
        cv2.imshow('Card Scanner - Live Feed', frame)
        cv2.imshow('Edge Detection', edged)

        # Break the loop when 'q' is pressed.
        if cv2.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture and destroy all windows.
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
