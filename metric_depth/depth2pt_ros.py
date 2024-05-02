# Born out of Issue 36. 
# Allows  the user to set up own test files to infer on (Create a folder my_test and add subfolder input and output in the metric_depth directory before running this script.)
# Make sure you have the necessary libraries
# Code by @1ssb

import argparse
import os
import glob
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import open3d as o3d
from tqdm import tqdm
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
import cv2
from util.transform import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose
from sensor_msgs.msg import PointCloud2, PointField
import rclpy
# Global settings
FL = 715.0873
cx, cy, FX, FY = 966.9719095357877, 521.7551265973169, 1452.5141034625492, 1443.1773620289302

NYU_DATA = False
FINAL_HEIGHT = 1080
FINAL_WIDTH = 1920
INPUT_DIR = './my_test/input'
OUTPUT_DIR = './my_test/output'
DATASET = 'nyu' # Lets not pick a fight with the model's dataloader

def process_images(model):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    image_paths = glob.glob(os.path.join(INPUT_DIR, '*.png')) + glob.glob(os.path.join(INPUT_DIR, '*.jpg'))
    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 5 != 0:
            continue

        color_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
        color_image = transform({'image': color_image})['image']
        # color_image = Image.open(image_path).convert('RGB')
        # original_width, original_height = color_image.shape[:2]
        image_tensor = torch.from_numpy(color_image).unsqueeze(0).to('cpu')#'cuda' if torch.cuda.is_available() else 'cpu')
        # image_tensor = transforms.ToTensor()(color_image).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')
        print(image_tensor.shape)
        pred = model(image_tensor, dataset=DATASET)
        if isinstance(pred, dict):
            pred = pred.get('metric_depth', pred.get('out'))
        elif isinstance(pred, (list, tuple)):
            pred = pred[-1]
        pred = pred.squeeze().detach().cpu().numpy()

        # Resize color image and depth to final size
        # resized_color_image = color_image.resize((int(FINAL_WIDTH), int(FINAL_HEIGHT)), cv2.INTER_CUBIC)
        resized_pred = Image.fromarray(pred).resize((FINAL_WIDTH, FINAL_HEIGHT), Image.NEAREST)

        focal_length_x, focal_length_y = (FX, FY) #
        x, y = np.meshgrid(np.arange(FINAL_WIDTH), np.arange(FINAL_HEIGHT))
        x = (x - cx) / focal_length_x
        y = (y - cy) / focal_length_y
        z = np.array(resized_pred)
        points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
        # colors = np.array(resized_color_image).reshape(-1, 3) / 255.0
        
        # Camera to ROS "z-up" conversion
        points_ros = np.zeros_like(points)
        points_ros[..., 0] = points[..., 2]  # ROS X = Camera Z
        points_ros[..., 1] = -points[..., 0]  # ROS Y = -Camera X
        points_ros[..., 2] = -points[..., 1]  # ROS Z = -Camera Y (assuming Y needs inversion)

        # Flatten points for PointCloud2
        points_flat = points_ros.reshape(-1, 3)

        # Add RGB to point cloud
        rgb_flat = frame.reshape(-1, 3)
        # Assuming your RGBD camera and ROS setup require specific handling for RGB data,
        # the following line combines XYZ and RGB into a single array for PointCloud2
        points_rgb_ros = np.concatenate([points_flat, rgb_flat], axis=1)

        # Create PointCloud2 message
        pt = PointCloud2()
        pt.header.frame_id = "map"
        # Timestamp and other header configurations
        pt.height = 1  # Unordered cloud
        pt.width = len(points_rgb_ros)
        pt.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            # Assuming RGB are in separate fields, adjust offsets and types as necessary
            PointField(name='r', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='g', offset=16, datatype=PointField.FLOAT32, count=1),
            PointField(name='b', offset=20, datatype=PointField.FLOAT32, count=1),
        ]
        pt.is_bigendian = False
        pt.point_step = 24  # Size of point + RGB in bytes
        pt.row_step = pt.point_step * pt.width
        pt.is_dense = True  # Assuming no invalid points
        pt.data = np.asarray(points_rgb_ros, np.float32).tobytes()

        # Publish the point cloud
        pt_pub.publish(pt)
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points)
        # print(frame.shape)
        # pcd.colors = o3d.utility.Vector3dVector(frame.reshape(-1, 3)/255)
        # o3d.io.write_point_cloud(os.path.join(OUTPUT_DIR, "frame.ply"), pcd)
    

def main(model_name, pretrained_resource):
    config = get_config(model_name, "eval", DATASET)
    config.pretrained_resource = pretrained_resource
    model = build_model(config).to('cpu')#'cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    process_images(model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default='zoedepth', help="Name of the model to test")
    parser.add_argument("-p", "--pretrained_resource", type=str, default='local::./checkpoints/depth_anything_metric_depth_indoor.pt', help="Pretrained resource to use for fetching weights.")
    rclpy.init()
    node = rclpy.create_node('point_cloud_publisher')
    pt_pub = node.create_publisher(PointCloud2, 'point_cloud', 10)
    args = parser.parse_args()  
    args = parser.parse_args()
    main(args.model, args.pretrained_resource)
