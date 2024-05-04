#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from matplotlib import pyplot as plt

from sensor_msgs.msg import PointCloud2, PointField, Image
import rclpy
import cv_bridge
default_width = 1920
default_height = 1080
cx, cy, fx, fy = 966.9719095357877, 521.7551265973169, 1452.5141034625492, 1443.1773620289302
transform_intrinsics = False

frame = None


def process_images(image, clone):
    global cx, cy, fx, fy, transform_intrinsics
    orig_height, orig_width = image.shape[:2]
    if not transform_intrinsics:
        cx, cy, fx, fy = cx * orig_width / default_width, cy * orig_height / default_height, fx * orig_width / default_width, fy * orig_height / default_height
        transform_intrinsics = True
    image = transform({'image': image})['image']
    frame_height, frame_width = image.shape[:2]
    image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        depth = depth_anything(image)

    depth = F.interpolate(depth[None], (orig_height, orig_width), mode='bilinear', align_corners=False)[0, 0]
    depth2pt(clone, depth)

def image_callback(msg):
    global frame, cx, cy, fx, fy, transform_intrinsics
    bridge = cv_bridge.CvBridge()
    frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    clone = frame.copy()
    process_images(frame, clone)

def depth2pt(image, depth):
        global cx, cy, fx, fy

        # Z = 1.40 - (depth.cpu().squeeze().numpy())/18
        Z = 20.5/depth.cpu().squeeze().numpy()
        # Z = depth.cpu().squeeze().numpy()
        height, width = depth.shape
        print("max depth: ", Z.max())
        print("min depth: ", Z.min())
        meshgrid = np.meshgrid(range(width), range(height), indexing='xy')
        # Convert depth to point cloud in camera coordinates
        points = np.stack(((meshgrid[0]-cx) * Z / fx, (meshgrid[1]-cy) * Z / fy, Z), axis=2)

        # Camera to ROS "z-up" conversion
        points_ros = np.zeros_like(points)
        points_ros[..., 0] = points[..., 2]  # ROS X = Camera Z
        points_ros[..., 1] = -points[..., 0]  # ROS Y = -Camera X
        points_ros[..., 2] = -points[..., 1]  # ROS Z = -Camera Y (assuming Y needs inversion)

        # Flatten points for PointCloud2
        points_flat = points_ros.reshape(-1, 3)

        # Add RGB to point cloud
        rgb_flat = image.reshape(-1, 3)

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
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl'])
    parser.add_argument('--src', type=str, default='cam', choices=['cam', 'topic'])
    rclpy.init()
    node = rclpy.create_node('point_cloud_publisher')
    pt_pub = node.create_publisher(PointCloud2, 'point_cloud', 10)
    image_sub = node.create_subscription(Image, 'image_raw', image_callback, 10)
    args = parser.parse_args()  
    
    margin_width = 50
    caption_height = 60

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(args.encoder)).to(DEVICE).eval()
    
    total_params = sum(param.numel() for param in depth_anything.parameters())
    print('Total parameters: {:.2f}M'.format(total_params / 1e6))
    
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

    if args.src == 'cam':
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = frame / 255.0
            clone = frame.copy()

            process_images(frame, clone)
            if cv2.waitKey(1) == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        rclpy.spin(node)
            

        rclpy.shutdown()
