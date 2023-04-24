import numpy as np
import cv2


def loadVid(path):
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(path)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    i = 0
    # Read until video is completed
    while (cap.isOpened()):
        # Capture frame-by-frame
        i += 1
        ret, frame = cap.read()
        if ret == True:

            # Store the resulting frame
            if i == 1:
                frames = frame[np.newaxis, ...]
            else:
                frame = frame[np.newaxis, ...]
                frames = np.vstack([frames, frame])
                frames = np.squeeze(frames)

        else:
            break

    # When everything done, release the video capture object
    cap.release()

    return frames


def trailer_dimensions(trailer_shape,cover_shape)->np.ndarray:
    # dimensions
    cover_w, cover_h = cover_shape[:2]
    trailer_w, trailer_h = trailer_shape[:2]

    # Get the corners of the cover
    corners = np.array([
            [0, 0],
            [0, cover_h-1],
            [cover_w-1, cover_h-1],
            [cover_w-1, 0]
        ],
        dtype=np.float32)

    # trailer dimensions wrt cover
    # Calculate dimensions
    x_min = int(np.min(corners[:,0]))
    x_max = int(np.max(corners[:,0]))
    y_min = int(np.min(corners[:,1]))
    y_max = int(np.max(corners[:,1]))

    # Calculate the dimensions of the new image
    new_w = min(trailer_w,int(x_max - x_min))
    new_h = min(trailer_h,int(y_max - y_min))

    # center point
    center = np.array([trailer_h/2, trailer_w/2])

    # trailer corners
    x_start = max(0, int(center[0] - new_h/2))
    x_end = min(trailer_h, int(center[0] + new_h/2))
    y_start = max(0, int(center[1] - new_w/2))
    y_end = min(trailer_w, int(center[1] + new_w/2))

    return np.array([x_start, x_end, y_start, y_end])

def prep_trailer_frame(trailer, homography, trailer_corners, cover_shape, book_shape)->np.ndarray:
    # x_start, x_end, y_start, y_end = trailer_corners

    cover_w, cover_h = cover_shape[:2]
    book_w, book_h = book_shape[:2]

    # cropped_trailer = trailer[x_start:x_end, y_start:y_end]
    cropped_trailer = cv2.resize(trailer, (cover_h, cover_w))
    cropped_trailer = wrap_prespective(cropped_trailer,homography,(book_h,book_w))
    return cropped_trailer

def overlay(cropped_trailer,book,mask, homography)->np.ndarray:
    # Create a mask of cover image and create its inverse mask also
    mask_3d = np.dstack([mask, mask, mask])
    mask = wrap_prespective(mask_3d,homography,(book.shape[1],book.shape[0]))[:,:,0]
    mask_inv = cv2.bitwise_not(mask)
    # Black-out the area of cover in book image
    book_bg = cv2.bitwise_and(book, book, mask=mask_inv)

    # Put cover in book image and modify the book image
    dst = cv2.add(book_bg, cropped_trailer)
    return dst

def out_frame(book, kp_cover, des_cover, trailer, cover_shape, mask):
    book_shape = book.shape[:2]

    book_gray = cv2.cvtColor(book, cv2.COLOR_BGR2GRAY)
    homography = cover_to_book_homography(kp_cover,des_cover, book_gray)

    # Get the trailer corners
    # trailer_corners = trailer_dimensions(trailer.shape, cover_shape)

    # Get the trailer frame
    cropped_trailer = prep_trailer_frame(trailer, homography, None, cover_shape, book_shape)

    # Overlay the trailer frame on the book
    dst = overlay(cropped_trailer, book, mask, homography)
    return dst


def transform_with_homography(h_mat, points_array):
    # add column of ones so that matrix multiplication with homography matrix is possible
    ones_col = np.ones((points_array.shape[0], 1))
    points_array = np.concatenate((points_array, ones_col), axis=1)
    transformed_points = np.matmul(h_mat, points_array.T)
    epsilon = 1e-7 # very small value to use it during normalization to avoid division by zero
    transformed_points = transformed_points / (transformed_points[2,:].reshape(1,-1) + epsilon)
    transformed_points = transformed_points[0:2,:].T

    return transformed_points