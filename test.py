import torch
import cv2
import gslic
import numpy as np
from skimage.segmentation import mark_boundaries

image = cv2.imread('r_11.png')

image_torch = torch.from_numpy(image).permute(2, 0, 1).to(torch.int)

index = torch.empty((image.shape[0], image.shape[1]), dtype=torch.int32)

start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
torch.cuda.synchronize()
start.record()
gslic.gslic(image_torch.flatten(), index.flatten(), image_torch.shape[1], image_torch.shape[2], 2000)
end.record()
torch.cuda.synchronize()
print(start.elapsed_time(end))

image_torch = image_torch.view(3, image_torch.shape[1], image_torch.shape[2])
index = index.view(image_torch.shape[1], image_torch.shape[2])
print(index.shape)

index = index.numpy()
# print(index)
image = image.astype(np.float32) / 255

marked_image = mark_boundaries(image, index)

cv2.imwrite('r_11_marked.png', (marked_image*255).astype(np.uint8))