import cv2
import torch
import fastcv
print(fastcv.__file__)
img = cv2.imread("artifacts/test.jpg")
img_tensor = torch.from_numpy(img).contiguous().cuda()
blurred_tensor = fastcv.GaussianBlur(img_tensor, 5, 1.0, 1.0)

blurred_image = blurred_tensor.cpu().numpy()
cv2.imwrite("output_gaussian_blur.jpg", blurred_image)

print("saved blurred image.")