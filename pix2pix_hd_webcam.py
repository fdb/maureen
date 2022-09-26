# This script loads a pretrained pix2pixHD model and uses it to run inference on a webcam feed.
import os

import cv2
import torchvision.transforms as transforms
import mediapipe as mp
import numpy as np

import pix2pixhd.util.util as util
from pix2pixhd.models.models import create_model
from pix2pixhd.options.test_options import TestOptions
from file_utils import ensure_directory

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


face_mesh_drawing_style = mp_drawing_styles.get_default_face_mesh_tesselation_style()
face_mesh_drawing_style.color = (255, 255, 255)
face_mesh_drawing_style.thickness = 1
face_mesh_drawing_style.circle_radius = 0

cv_window_name = "window"
cv2.namedWindow(cv_window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(cv_window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

DRAW_METHOD_CAMERA = "cam"
DRAW_METHOD_CAMERA_AND_MESH = "cam+mesh"
DRAW_METHOD_CAMERA_AND_MESH_OVERLAY = "cam+mesh_overlay"
DRAW_METHOD_MESH_AND_GENERATED = "mesh+gen"
DRAW_METHOD_CAMERA_AND_GENERATED = "cam+gen"
DRAW_METHOD_GENERATED = "gen"
DRAW_METHODS = [
    DRAW_METHOD_CAMERA,
    DRAW_METHOD_CAMERA_AND_MESH,
    DRAW_METHOD_CAMERA_AND_MESH_OVERLAY,
    DRAW_METHOD_MESH_AND_GENERATED,
    DRAW_METHOD_CAMERA_AND_GENERATED,
    DRAW_METHOD_GENERATED,
]
draw_method = DRAW_METHOD_CAMERA


def _image_centered(img):
    dx = (1920 - 1024) // 2
    dy = (1080 - 1024) // 2
    img = np.pad(
        img,
        ((dy, 1080 - 1024 - dy), (dx, 1920 - 1024 - dx), (0, 0)),
        "constant",
        constant_values=0,
    )
    return img


def _image_side_by_side(img_left, img_right):
    img_left = cv2.resize(img_left, (960, 960))
    img_right = cv2.resize(img_right, (960, 960))
    img = np.concatenate((img_left, img_right), axis=1)
    # Center the generated image in a 1920x1080 image
    dy = (1080 - 960) // 2
    img = np.pad(
        img,
        ((dy, dy), (0, 0), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    return img


options = TestOptions()
options.parser.add_argument(
    "--method",
    type=str,
    help="Method to use: tesselation or contours",
    default="tesselation",
)
opt = options.parse(save=False)
opt.nThreads = 1
opt.batchSize = 1
opt.no_flip = True
opt.no_instance = True
opt.label_nc = 0

method = "contours"
connections = mp_face_mesh.FACEMESH_TESSELATION
if opt.method == "contours":
    connections = mp_face_mesh.FACEMESH_CONTOURS

model = create_model(opt)


transform_list = []
transform_list.append(transforms.ToTensor())
# transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
transform = transforms.Compose(transform_list)

# Create a VideoCapture object
cap = cv2.VideoCapture(0)
# By default, camera is set to 640x480 resolution.
# Setting a higher resolution doesn't seem to improve quality.
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.2,
    min_tracking_confidence=0.2,
) as face_mesh:
    while cap.isOpened():
        # Capture frame-by-frame
        ret, cam_image = cap.read()

        cam_image.flags.writeable = False

        # Flip the frame
        cam_image = cv2.flip(cam_image, 1)

        # Resize the image to the new dimensions.
        (h, w, _) = cam_image.shape
        new_height = 1024
        new_width = int(w * new_height / h)
        cam_image = cv2.resize(cam_image, (new_width, new_height))

        # Crop out a square of 1024x1024 pixels
        dx = (new_width - new_height) // 2
        dy = 0
        cam_image = cam_image[:, dx : dx + new_height]

        results = face_mesh.process(cam_image)

        mesh_image = np.zeros((1024, 1024, 3), np.uint8)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=mesh_image,
                    landmark_list=face_landmarks,
                    connections=connections,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=face_mesh_drawing_style,
                )

        # Transform to a tensor
        a_tensor = transform(mesh_image)
        # Run the model
        generated = model.inference(a_tensor, a_tensor)
        # Convert the tensor to an image
        generated_image = generated.data.cpu().float().numpy()
        # Un-normalize
        generated_image = (np.transpose(generated_image, (1, 2, 0)) + 1) / 2.0 * 255.0
        # Convert to uint8
        generated_image = generated_image.astype(np.uint8)
        # Flip color information
        generated_image = cv2.cvtColor(generated_image, cv2.COLOR_BGR2RGB)

        if draw_method == DRAW_METHOD_CAMERA:
            final_image = _image_centered(cam_image)
        elif draw_method == DRAW_METHOD_CAMERA_AND_MESH:
            final_image = _image_side_by_side(cam_image, mesh_image)
        elif draw_method == DRAW_METHOD_CAMERA_AND_MESH_OVERLAY:
            annotated_image = cam_image.copy()
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=annotated_image,
                        landmark_list=face_landmarks,
                        connections=connections,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=face_mesh_drawing_style,
                    )
            final_image = _image_centered(annotated_image)
        elif draw_method == DRAW_METHOD_MESH_AND_GENERATED:
            final_image = _image_side_by_side(mesh_image, generated_image)
        elif draw_method == DRAW_METHOD_CAMERA_AND_GENERATED:
            final_image = _image_side_by_side(cam_image, generated_image)
        elif draw_method == DRAW_METHOD_GENERATED:
            final_image = _image_centered(generated_image)

        # Display the resulting frame
        cv2.imshow(cv_window_name, final_image)

        # Press S to save a frame
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            ensure_directory("captures")
            # Find the last saved frame, in the format frame-0000.png
            last_frame = 0
            for filename in os.listdir("captures"):
                if filename.startswith("frame-") or filename.startswith("generated-"):
                    last_frame = max(last_frame, int(filename[6:10]))
            # Save the frame
            cv2.imwrite(
                "captures/final-%04d.png" % (last_frame + 1),
                final_image,
            )
            cv2.imwrite(
                "captures/generated-%04d.png" % (last_frame + 1),
                generated_image,
            )

        # Press D to switch drawing methods
        if key == ord("1"):
            draw_method = DRAW_METHOD_CAMERA
        elif key == ord("2"):
            draw_method = DRAW_METHOD_CAMERA_AND_MESH
        elif key == ord("3"):
            draw_method = DRAW_METHOD_CAMERA_AND_MESH_OVERLAY
        elif key == ord("4"):
            draw_method = DRAW_METHOD_MESH_AND_GENERATED
        elif key == ord("5"):
            draw_method = DRAW_METHOD_CAMERA_AND_GENERATED
        elif key == ord("6"):
            draw_method = DRAW_METHOD_GENERATED

        # Press Q on keyboard to exit
        if key == ord("q"):
            break
