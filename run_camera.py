import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', type=str)
    parser.add_argument('--outdir', type=str, default='./vis_video_depth')
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl'])
    
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

    cap = cv2.VideoCapture(0)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 4 == 0:
            continue

        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
        # resize the frame to the desired size multiple of 14 for the model
        frame = cv2.resize(frame, (518, 518), interpolation=cv2.INTER_CUBIC)
        frame_height, frame_width = frame.shape[:2]
        print(frame.shape)
        # frame = transform({'frame': frame})['frame']
        frame = torch.from_numpy(frame).unsqueeze(0).to(DEVICE)
        frame = frame.permute(0, 3, 1, 2)
        frame = frame.float()
        print(frame.shape)
        with torch.no_grad():
            depth = depth_anything(frame)

        depth = F.interpolate(depth[None], (frame_height, frame_width), mode='bilinear', align_corners=False)[0, 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        
        depth = depth.cpu().numpy().astype(np.uint8)
        depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
        
        split_region = np.ones((frame_height, margin_width, 3), dtype=np.float32) * 255
        frame_np = frame.squeeze().permute(1, 2, 0).cpu().numpy()
        depth_np = depth_color.astype(np.float32) / 255.0
        print("frame_np shape", frame_np.shape, "type", frame_np.dtype)
        print("depth_np shape", depth_np.shape, "type", depth_np.dtype)
        print("split_region shape", split_region.shape, "type", split_region.dtype)
        combined_frame = cv2.hconcat([frame_np, split_region, depth_np])
        caption_space = np.ones((caption_height, combined_frame.shape[1], 3), dtype=np.float32) * 255
        captions = ['Raw image', 'Depth Anything']
        segment_width = frame.shape[1] + margin_width
        for i, caption in enumerate(captions):
            # Calculate text size
            text_size = cv2.getTextSize(caption, font, font_scale, font_thickness)[0]

            # Calculate x-coordinate to center the text
            text_x = int((segment_width * i) + (frame.shape[1] - text_size[0]) / 2)

            # Add text caption
            cv2.putText(caption_space, caption, (text_x, 40), font, font_scale, (0, 0, 0), font_thickness)

        final_result = cv2.vconcat([caption_space, combined_frame])

        cv2.imshow('Depth Anything', final_result)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()


