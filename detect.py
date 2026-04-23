import cv2
import os
from ultralytics import YOLO

MODEL_PATH = 'best_license_plate.pt'
VIDEO_PATH = 'sample2.mp4'
OUTPUT_DIR = 'output'
CONFIDENCE_THRESHOLD = 0.3

model = YOLO(MODEL_PATH)

results = model.predict(
    source=VIDEO_PATH,
    conf=CONFIDENCE_THRESHOLD,
    save=True,
    save_txt=False,
    project=OUTPUT_DIR,
    name='predict',
    stream=False,
    show=False
)

output_folder = os.path.join(OUTPUT_DIR, 'predict')
output_video_path = None

for file in os.listdir(output_folder):
    if file.endswith('.mp4'):
        output_video_path = os.path.join(output_folder, file)
        break
    elif file.endswith('.avi'):
        avi_path = os.path.join(output_folder, file)
        mp4_path = avi_path.replace('.avi', '.mp4')

        cap = cv2.VideoCapture(avi_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = cv2.VideoWriter(mp4_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        cap.release()
        out.release()

        output_video_path = mp4_path
        break

if output_video_path:
    print("Inference complete! Output saved to:", os.path.abspath(output_video_path))
else:
    print("No output video found. Something may have gone wrong.")
