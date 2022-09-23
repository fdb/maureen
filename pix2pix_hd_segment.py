from glob import glob
import os
import cv2
import numpy as np
import mediapipe as mp
from file_utils import ensure_directory
import argparse
from glob import glob

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh_drawing_style = mp_drawing_styles.get_default_face_mesh_tesselation_style()
face_mesh_drawing_style.color = (255, 255, 255)
face_mesh_drawing_style.thickness = 1
face_mesh_drawing_style.circle_radius = 0
print(face_mesh_drawing_style)

def convert(input_folder, output_folder, method="tesselation"):
    output_folder_a = os.path.join(output_folder, "train_A")
    output_folder_b = os.path.join(output_folder, "train_B")
    ensure_directory(output_folder_a)
    ensure_directory(output_folder_b)

    connections = mp_face_mesh.FACEMESH_TESSELATION
    if method == "contours":
        connections = mp_face_mesh.FACEMESH_CONTOURS

    input_files = glob(os.path.join(input_folder, "*.jpeg"))
    face_index = 1
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        for input_file in input_files:
            image = cv2.imread(input_file)

            (h, w, _) = image.shape
            new_height = 1024
            new_width = int(w * new_height / h)
            # Resize the image to the new dimensions.
            image = cv2.resize(image, (new_width, new_height))

            # Crop out a square of 1024x1024 pixels
            dx = (new_width - new_height) // 2
            dy = 0
            image = image[:, dx : dx + new_height]

            # Convert the BGR image to RGB before processing.
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if not results.multi_face_landmarks:
                continue

            annotated_image = np.zeros((1024, 1024, 3), np.uint8)

            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=connections,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=face_mesh_drawing_style,
                )

            # Save the two images in train_A and train_B folders.
            # We're training from A to B, so from the annotated image to the source image.
            cv2.imwrite(os.path.join(output_folder_a, f"{face_index:06d}.png"), annotated_image)
            cv2.imwrite(os.path.join(output_folder_b, f"{face_index:06d}.png"), image)

            face_index += 1
            if face_index % 10 == 0:
                print(".", end="", flush=True)
    print()

parser = argparse.ArgumentParser(description='Convert video to segmented images')
parser.add_argument('input_folder', type=str, help='Input folder')
parser.add_argument('output_folder', type=str, help='Output folder')
parser.add_argument('--method', type=str, help='Method to use: tesselation or contours', default='tesselation')
args = parser.parse_args()
convert(args.input_folder, args.output_folder, args.method)

