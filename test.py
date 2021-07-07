#!/usr/bin/env python
import dlib
import cv2
import numpy as np
from draw_annotation_box import draw_annotation_box
import math

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

def determine_facemarks (image, detector, predictor):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    if len(rects) != 1:
        return None, None, False
    else:
        shape = predictor(gray, rects[0])
        shape = shape_to_np(shape)
        return shape, rects[0], True


def apply_facemarks (image, shape, color=(0, 255, 0)):
    marked_images = image.copy()
    for (x, y) in shape:
        cv2.circle(marked_images, (x, y), 3, color, -1)

    return marked_images

def create_facestuff(image):
    # Read Image
    im = cv2.imread(image)
    normed_image = resize(im, width = 320)

    predictor_path = 'shape_predictor_81_face_landmarks.dat'

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    fm, fm_rect, fm_ok = determine_facemarks(normed_image, detector, predictor)
    if len(fm) == 81:
        fm = np.take(fm, [30,8,36,45,48,54], 0)
    marked_images_image = apply_facemarks(normed_image, fm, (0,0,255))

    # cv2.imshow("test", marked_images_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    size = normed_image.shape


    #2D image points. Changes for each image
    image_points = np.float32(fm)

    # 3D model points.
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner

    ])

    # Camera internals

    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype = "float32")

    print( "Camera Matrix :\n {0}".format(camera_matrix))

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    print("Rotation Vector:\n {0}".format(rotation_vector))
    print("Translation Vector:\n {0}".format(translation_vector))

    # Project a 3D point (0, 0, 1000.0) onto the image plane.
    # We use this to draw a line sticking out of the nose

    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    for p in image_points:
        cv2.circle(normed_image, (int(p[0]), int(p[1])), 3, (0,0,255), -1)

    p1 = ( int(image_points[0][0]), int(image_points[0][1]))
    p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

    # cv2.line(normed_image, p1, p2, (255,0,0), 2)

    # Annotation box
    x1, x2 = draw_annotation_box(normed_image, rotation_vector, translation_vector, camera_matrix)

    # cv2.line(normed_image, p1, p2, (0, 255, 255), 2)
    cv2.line(normed_image, tuple(x1), tuple(x2), (255, 255, 0), 2)
    # for (x, y) in shape:
    #     cv2.circle(img, (x, y), 4, (255, 255, 0), -1)
    # cv2.putText(img, str(p1), p1, font, 1, (0, 255, 255), 1)
    # try:
    #     m = (p2[1] - p1[1])/(p2[0] - p1[0])
    #     ang1 = int(math.degrees(math.atan(m)))
    # except:
    #     ang1 = 90

    try:
        m = (x2[1] - x1[1])/(x2[0] - x1[0])
        ang2 = int(math.degrees(math.atan(-1/m)))
    except:
        ang2 = 90

    # cv2.putText(normed_image, str(ang1), tuple(p1), cv2.FONT_HERSHEY_SIMPLEX , 2, (128, 255, 255), 3)
    cv2.putText(normed_image, str(ang2), tuple(x1), cv2.FONT_HERSHEY_SIMPLEX , 2, (255, 255, 128), 3)

    # Display image
    cv2.imshow("Output", normed_image)


# Add Photos to List to show head pose angle
for i in ['1.jpeg', '2.jpeg', '3.jpeg', '4.jpeg', '5.jpeg', '6.jpeg', '7.jpeg']:
    create_facestuff(i)
    cv2.waitKey(0)