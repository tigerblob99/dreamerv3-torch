import numpy as np
import cv2

data = np.load('image.npy', allow_pickle=True)

# Define output path
output_path = 'output_video_256.mp4'

height, width = 256, 256
fps = 20
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

for frame in data:
    # Resize from 64×64 → 512×512
    up = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)
    out.write(cv2.cvtColor(up, cv2.COLOR_RGB2BGR))

out.release()
output_path


