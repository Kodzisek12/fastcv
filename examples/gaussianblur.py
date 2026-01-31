import cv2
import torch
import fastcv

img = cv2.imread("artifacts/test.jpg")
img_tensor = torch.from_numpy(img).cuda()
blurred_tensor = fastcv.GaussianBlur(img_tensor, 15, 5, 5)

blurred_image = blurred_tensor.cpu().numpy()
cv2.imwrite("output__gaussian_blur.jpg", blurred_image)

print("saved blurred image.")