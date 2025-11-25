import cv2
import mediapipe as mp
import argparse
import os
import json
from datetime import datetime

def setup_mediapipe():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return pose

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

    pose = setup_mediapipe()
    
    frames_data = []
    
    frame_idx = 0
    processed_count = 0
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        if frame_idx % sample_rate == 0:
            # Convert the BGR image to RGB.
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
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
            # Create output filename: video_name.json
            output_filename = os.path.splitext(video_file)[0] + ".json"
            output_path = os.path.join(output_dir, output_filename)
            process_video(video_path, output_path, args.sample_rate)
            
    else:
        # Single file processing
        if args.output:
            output_path = args.output
        else:
            # Default output name if not specified
            output_path = os.path.splitext(input_path)[0] + ".json"
            
        process_video(input_path, output_path, args.sample_rate)

if __name__ == "__main__":
    main()
