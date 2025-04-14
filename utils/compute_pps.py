import cv2
from PIL import Image
import numpy as np
import os
import glob
import torch
import torch.nn.functional as F
import matplotlib.colors as mcolors
import time


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gamma = 1 / 2.2

K = [227.60416, 237.5, 227.60416, 237.5]
fx, fy, cx, cy = K[0], K[1], K[2], K[3]

def compute_pps(depth, scale, fx, fy, cx, cy):
    normals, positions = compute_normals_from_depth(depth, fx, fy, cx, cy) 
    L  = torch.zeros((3)).unsqueeze(1).unsqueeze(1) - positions # surface to camera rays [3, H, W]

    distance = L.norm(p=2, dim=0, keepdim=True).square()
    A = 1 / distance
    A = A / A.max()
    
    L = F.normalize(L, dim=0)
    
    COS = (normals * L).sum(dim=0, keepdim=True).clamp(min=-1, max=1)
    PPS = A * COS
    return L, normals, PPS

def compute_albedo(color):
    hsv_vis = mcolors.rgb_to_hsv(color)
    hsv_albedo = np.copy(hsv_vis)
    hsv_albedo[:, :, 2] = 1.0  # Set V to 100% brightness
    return mcolors.hsv_to_rgb(hsv_albedo)

def compute_normals_from_depth(depth, fx, fy, cx, cy):
    """
    Compute surface normals from the depth map and return 3D points X.
    Args:
        depth (torch.Tensor): Depth map of shape (1, H, W).
        intrinsics (torch.Tensor): Camera intrinsic matrix of shape (3, 3).
    Returns:
        normals (torch.Tensor): Surface normals of shape (3, H, W).
        X (torch.Tensor): 3D points of shape (3, H, W).
    """
    _, height, width = depth.shape
    device = depth.device
    # Create a mesh grid of pixel coordinates
    y, x = torch.meshgrid(torch.arange(0, height, device=device),
                          torch.arange(0, width, device=device))
    
    # Unproject to 3D points
    Z = depth
    X = (x.float() - cx) * Z / fx
    Y = (y.float() - cy) * Z / fy
    
    # Stack to get 3D positions of shape (3, H, W)
    positions = torch.cat([X, Y, Z], dim=0)
    P_u = torch.gradient(positions, dim=-1)[0]
    P_v = torch.gradient(positions, dim=-2)[0]


    normals = torch.cross(P_v, P_u, dim=0)
    normals = F.normalize(normals, dim=0)
    return normals, positions

def process_depth_image(args):
    """Process a single depth image and save its PPS output."""
    depth_file, input_dir, output_dir = args  # Unpack tuple

    try:
        # Construct the output file path
        relative_path = os.path.relpath(depth_file, input_dir)
        output_file = os.path.join(output_dir, relative_path).replace("depths", "PPS")

        if os.path.isfile(output_file):
            print(f"Skipping {depth_file}, already exists")
            return  

        print(f"Processing {depth_file}")  # Debugging print

        # Load the depth image
        depth_png = np.array(Image.open(depth_file)) / 256.0 / 255.0

        # Convert to torch tensor for processing
        depth_png = torch.tensor(depth_png, dtype=torch.float32).unsqueeze(0)

        # Compute PPS
        _, _, PPS = compute_pps(depth_png, 1.0, fx, fy, cx, cy)
        PPS = PPS.numpy().squeeze()
        # PPS = remove_depth_valleys_v3(PPS**gamma, threshold=0.8, dilation_iter=3)
        # PPS = guided_filter(PPS, PPS, r=10, eps=1e-3)

        # Create the directory for the output file if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Save the processed image
        Image.fromarray((PPS * 255).astype(np.uint8)).save(output_file)
        print(f"Processed: {output_file}")  # Debugging print

    except Exception as e:
        print(f"Error processing {depth_file}: {e}")


if __name__ == "__main__":
    input_dir = "/playpen-nas-ssd/akshay/3D_medical_vision/datasets/SimCol3D_TANDEM_Format"
    output_dir = "/playpen-nas-ssd3/anaxxq/dataset/SimCol"

    process_images_parallel(input_dir, output_dir, num_workers=8)

    # Final check: List the first few output images to verify
    output_files = glob.glob(os.path.join(output_dir, '**', 'PPS', '*.png'), recursive=True)
    print("Sample output files:", output_files[:5])