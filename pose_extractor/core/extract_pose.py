import cv2
import mediapipe as mp
import argparse
import os
import json
import numpy as np
import subprocess
import shutil
from datetime import datetime

def setup_mediapipe(static_image_mode=False, min_confidence=0.7):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=static_image_mode,
        model_complexity=2,
        enable_segmentation=False,
        smooth_landmarks=True,  # Enable temporal smoothing
        min_detection_confidence=min_confidence,
        min_tracking_confidence=min_confidence
    )
    return pose

def convert_to_720p(input_path, output_path):
    """
    Converts video to 720p using ffmpeg with high quality settings
    to preserve details for pose estimation.
    """
    if not shutil.which('ffmpeg'):
        print("Error: ffmpeg is not installed or not in PATH.")
        return False

    print(f"Converting {input_path} to 720p (High Quality)...")
    
    # High quality settings:
    # - scale=-2:720: Keep aspect ratio, height 720, width divisible by 2
    # - flags=lanczos: High quality scaling algorithm (sharper than default)
    # - crf 18: High quality (lower is better, 18 is roughly visually lossless)
    # - preset slow: Better compression efficiency
    # - pix_fmt yuv420p: Ensure compatibility
    command = [
        'ffmpeg', '-i', input_path,
        '-vf', 'scale=-2:720:flags=lanczos',
        '-c:v', 'libx264',
        '-crf', '18',
        '-preset', 'slow',
        '-c:a', 'copy',
        '-pix_fmt', 'yuv420p',
        output_path,
        '-y' # Overwrite if exists
    ]
    
    try:
        # Run ffmpeg silently (stdout/stderr to PIPE) unless error
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Conversion complete: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting video: {e}")
        return False

def process_video(video_path, output_path, sample_rate=1):
    print(f"Processing video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video Info: {width}x{height} @ {fps}fps, {frame_count} frames")

    # Use video mode (static_image_mode=False) if processing every frame (sample_rate=1)
    # This enables temporal smoothing and tracking, reducing jitter and error.
    use_video_mode = (sample_rate == 1)
    if use_video_mode:
        print("Mode: Video Stream (Smoothing Enabled)")
    else:
        print(f"Mode: Static Images (Sample Rate: {sample_rate})")

    pose = setup_mediapipe(static_image_mode=not use_video_mode, min_confidence=0.7)
    
    frames_data = []
    
    frame_idx = 0
    processed_count = 0
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        if frame_idx % sample_rate == 0:
            # Enhance image for better detection on compressed videos
            # 1. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl,a,b))
            image_enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

            # 2. Mild Sharpening to recover edges lost in compression
            kernel = np.array([[0, -1, 0],
                               [-1, 5,-1],
                               [0, -1, 0]])
            image_enhanced = cv2.filter2D(image_enhanced, -1, kernel)

            # Convert the BGR image to RGB.
            image_rgb = cv2.cvtColor(image_enhanced, cv2.COLOR_BGR2RGB)
            
            # Process
            results = pose.process(image_rgb)
            
            if results.pose_landmarks:
                landmarks = []
                for i, lm in enumerate(results.pose_landmarks.landmark):
                    landmarks.append({
                        "id": i,
                        "name": mp.solutions.pose.PoseLandmark(i).name,
                        "x": round(lm.x, 6),
                        "y": round(lm.y, 6),
                        "z": round(lm.z, 6),
                        "visibility": round(lm.visibility, 6)
                    })
                
                world_landmarks = []
                if results.pose_world_landmarks:
                    for i, lm in enumerate(results.pose_world_landmarks.landmark):
                        world_landmarks.append({
                            "id": i,
                            "name": mp.solutions.pose.PoseLandmark(i).name,
                            "x": round(lm.x, 6),
                            "y": round(lm.y, 6),
                            "z": round(lm.z, 6),
                            "visibility": round(lm.visibility, 6)
                        })
                
                frames_data.append({
                    "frame_index": frame_idx,
                    "timestamp_sec": round(frame_idx / fps, 3),
                    "landmarks": landmarks,
                    "world_landmarks": world_landmarks
                })
                processed_count += 1
                print(f"Processed frame {frame_idx}/{frame_count}", end='\r')

        frame_idx += 1

    cap.release()
    print(f"\nFinished processing. Extracted {processed_count} frames.")
    
    # Construct full data object
    full_data = {
        "video_info": {
            "filename": os.path.basename(video_path),
            "width": width,
            "height": height,
            "fps": fps,
            "total_frames": frame_count,
            "extracted_frames": len(frames_data),
            "processed_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "frames": frames_data
    }

    save_output(output_path, full_data)

def save_output(output_path, data):
    ext = os.path.splitext(output_path)[1].lower()
    
    if ext == '.json':
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"JSON data saved to: {output_path}")
    else:
        # Default to Markdown
        generate_markdown(output_path, data)

def generate_markdown(output_path, data):
    info = data['video_info']
    frames = data['frames']
    
    md_content = f"""# Pose Data Extraction Report

## Video Information
- **File Name:** `{info['filename']}`
- **Resolution:** {info['width']}x{info['height']}
- **FPS:** {info['fps']}
- **Total Frames:** {info['total_frames']}
- **Extracted Frames:** {info['extracted_frames']}
- **Processed Date:** {info['processed_date']}

## Keyframe Data (Summary)

Only showing first 5 landmarks (Nose, Eyes, Ears) for brevity in table. Full data in JSON section.

"""
    
    for frame in frames:
        md_content += f"### Frame {frame['frame_index']} (Time: {frame['timestamp_sec']}s)\n\n"
        md_content += "| ID | Name | X | Y | Z | Visibility |\n"
        md_content += "|---|---|---|---|---|---|\n"
        
        # Show only first 11 landmarks (Head/Face) to keep MD readable, or maybe key body parts?
        # Let's show a few key points: 0(Nose), 11(Left Shoulder), 12(Right Shoulder), 23(Left Hip), 24(Right Hip), 25(Left Knee), 26(Right Knee), 27(Left Ankle), 28(Right Ankle)
        key_points = [0, 11, 12, 23, 24, 25, 26, 27, 28]
        
        for lm in frame['landmarks']:
            if lm['id'] in key_points:
                md_content += f"| {lm['id']} | {lm['name']} | {lm['x']} | {lm['y']} | {lm['z']} | {lm['visibility']} |\n"
        
        md_content += "\n"

    md_content += """
## Full Data (JSON)

```json
"""
    md_content += json.dumps(data, indent=2)
    md_content += "\n```\n"

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"Markdown report saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Extract Pose Landmarks from Video to Markdown')
    parser.add_argument('input_path', help='Path to the input video file or directory containing mp4 files')
    parser.add_argument('--output', '-o', help='Path to the output markdown file or directory', default=None)
    parser.add_argument('--sample_rate', '-s', type=int, default=1, help='Process every Nth frame (default: 1)')
    parser.add_argument('--convert-720p', action='store_true', help='Automatically convert video to 720p HQ before processing')
    
    args = parser.parse_args()
    
    input_path = args.input_path
    
    if os.path.isdir(input_path):
        # Batch processing for directory
        video_files = [f for f in os.listdir(input_path) if f.lower().endswith('.mp4')]
        if not video_files:
            print(f"No .mp4 files found in {input_path}")
            return
            
        output_dir = args.output if args.output else input_path
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        print(f"Found {len(video_files)} videos in directory. Processing...")
        
        for video_file in video_files:
            video_path = os.path.join(input_path, video_file)
            
            # Handle conversion if requested
            if args.convert_720p:
                converted_filename = os.path.splitext(video_file)[0] + "_720p_hq.mp4"
                converted_path = os.path.join(output_dir, converted_filename)
                if convert_to_720p(video_path, converted_path):
                    video_path = converted_path # Use the converted video for processing
                    # Update output filename to match original name but with json extension
                    # or keep the _720p suffix if preferred. Let's keep it clean:
                    output_filename = os.path.splitext(video_file)[0] + ".json"
                else:
                    print(f"Skipping {video_file} due to conversion error.")
                    continue
            else:
                output_filename = os.path.splitext(video_file)[0] + ".json"

            output_path = os.path.join(output_dir, output_filename)
            process_video(video_path, output_path, args.sample_rate)
            
    else:
        # Single file processing
        video_path = input_path
        
        if args.output:
            # If output is a directory
            if os.path.isdir(args.output) or (not os.path.splitext(args.output)[1]):
                 if not os.path.exists(args.output):
                     os.makedirs(args.output)
                 output_dir = args.output
                 output_filename = os.path.splitext(os.path.basename(input_path))[0] + ".json"
                 output_path = os.path.join(output_dir, output_filename)
            else:
                 output_path = args.output
                 output_dir = os.path.dirname(output_path)
        else:
            # Default output name if not specified
            output_dir = os.path.dirname(input_path)
            output_path = os.path.splitext(input_path)[0] + ".json"

        if args.convert_720p:
            # Determine where to save the converted video
            # Save it in the same folder as output or input
            save_dir = output_dir if output_dir else os.path.dirname(input_path)
            converted_filename = os.path.splitext(os.path.basename(input_path))[0] + "_720p_hq.mp4"
            converted_path = os.path.join(save_dir, converted_filename)
            
            if convert_to_720p(video_path, converted_path):
                video_path = converted_path
            else:
                print("Conversion failed, aborting.")
                return
            
        process_video(video_path, output_path, args.sample_rate)

if __name__ == "__main__":
    main()
