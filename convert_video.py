import cv2
import numpy as np
import mediapipe as mp
from glob import glob
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# For static images:
IMAGE_VIDEO = "input_video/maureen_2.mp4"
# Import the video as separate frames
cap = cv2.VideoCapture(IMAGE_VIDEO)


face_index = 1
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
) as face_mesh:
    while True:
        print(face_index)
        ret, frame = cap.read()
        # Bail out if we don't have a frame
        if not ret:
            break

        # for idx, file in enumerate(IMAGE_FILES):
        # image = cv2.imread(file)
        image = frame
        (h, w, _) = image.shape
        # factor = 512/w
        new_height = 512
        new_width = int(w * new_height / h)
        # Make it smaller
        image = cv2.resize(image, (new_width, new_height))

        # Crop out a square of 512x512 out of the video
        dx = (new_width - 512) // 2
        dy = 0
        image = image[:, dx : dx + 512]

        # Convert the BGR image to RGB before processing.
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print and draw face mesh landmarks on the image.
        if not results.multi_face_landmarks:
            continue
        # annotated_image = image.copy()
        # Fill the entire image with white
        # print(image.shape)
        # (width, height, depth) = image.shape
        annotated_image = np.zeros((512, 512, 3), np.uint8)
        # size = cv2.GetSize(image)
        # annotated_image = cv2.createImage(size, cv2.IPL_DEPTH_8U, 3)

        for face_landmarks in results.multi_face_landmarks:
            # print("face_landmarks:", face_landmarks)
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
            )
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
            )
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
            )

        # Combine the two images
        combined_image = np.zeros((512, 1024, 3), np.uint8)
        combined_image[:, 0:512] = image
        combined_image[:, 512:1024] = annotated_image

        # output_fname = os.path.join("output_video", os.path.basename(file))
        output_fname = os.path.join("output_video_2", f"face-{face_index:04}.jpeg")

        cv2.imwrite(output_fname, combined_image)

        face_index += 1
