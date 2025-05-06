from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch
import depth_pro

# Load model and preprocessing transform
model, transform = depth_pro.create_model_and_transforms()
model.eval()

# Load and preprocess an image
image, _, f_px = depth_pro.load_rgb("../data/hos.jpeg")

# Show original image
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')

# Apply the transform
transformed_image = transform(image)

# Convert transformed image tensor to displayable format
def tensor_to_pil(img_tensor):
    # Unnormalize based on assumed [-1, 1] normalization (change if needed)
    unnormalize = T.Normalize(
        mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
        std=[1 / 0.5, 1 / 0.5, 1 / 0.5]
    )
    img_tensor = unnormalize(img_tensor)
    img_tensor = img_tensor.clamp(0, 1)
    return T.ToPILImage()(img_tensor)

# Show transformed image
plt.subplot(1, 3, 2)
plt.imshow(tensor_to_pil(transformed_image))
plt.title("Transformed Image")
plt.axis('off')

# Run inference
with torch.no_grad():
    prediction = model.infer(transformed_image, f_px=f_px)
depth = prediction["depth"]
focallength_px = prediction["focallength_px"]

# Print depth map info
print("Depth shape:", depth.shape)
print("Depth values:\n", depth.squeeze().cpu().numpy())

# Visualize depth map
depth_np = depth.squeeze().cpu().numpy()
plt.subplot(1, 3, 3)
plt.imshow(depth_np, cmap='plasma')
plt.colorbar(label='Depth (m)')
plt.title("Predicted Depth Map")
plt.axis('off')

plt.tight_layout()
plt.show()

# -----

# import torch
# import depth_pro
# import numpy as np

# # Load model and transform
# model, transform = depth_pro.create_model_and_transforms()
# model.eval()

# # Load image and focal length
# image, _, f_px = depth_pro.load_rgb("../data/example.jpg")
# image = transform(image)

# # Inference
# with torch.no_grad():
#     prediction = model.infer(image, f_px=f_px)

# # Get depth in meters
# depth = prediction["depth"]
# depth_np = depth.squeeze().cpu().numpy()

# # Print real-world depth values
# print("Depth shape:", depth_np.shape)
# print("Min depth (m):", np.min(depth_np))
# print("Max depth (m):", np.max(depth_np))
# print("Depth values (in meters):\n", depth_np)



# # from PIL import Image
# # import depth_proimport torch
# import depth_pro
# import numpy as np

# # Load model and transform
# model, transform = depth_pro.create_model_and_transforms()
# model.eval()

# # Load image and focal length
# image, _, f_px = depth_pro.load_rgb("../data/example.jpg")
# image = transform(image)

# # Inference
# with torch.no_grad():
#     prediction = model.infer(image, f_px=f_px)

# # Get depth in meters
# depth = prediction["depth"]
# depth_np = depth.squeeze().cpu().numpy()

# # Print real-world depth values
# print("Depth shape:", depth_np.shape)
# print("Min depth (m):", np.min(depth_np))
# print("Max depth (m):", np.max(depth_np))
# print("Depth values (in meters):\n", depth_np)


# # Load model and preprocessing transform
# model, transform = depth_pro.create_model_and_transforms()
# model.eval()

# # Load and preprocess an image.
# image, _, f_px = depth_pro.load_rgb("../data/example.jpg")
# image = transform(image)

# # Run inference.
# prediction = model.infer(image, f_px=f_px)
# depth = prediction["depth"]  # Depth in [m].
# focallength_px = prediction["focallength_px"]  # Focal length in pixels.