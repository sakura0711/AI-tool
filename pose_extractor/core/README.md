# Pose Extractor Tool

這是一個用於從影片中提取 MediaPipe Pose 關鍵點並保存為 **JSON** 或 Markdown 格式的 Python 工具。
此工具是為了協助建立「標準動作數據」而設計，輸出的數據可用於 Phase 3 的姿勢比對系統。

## 安裝

1. 確保已安裝 Python 3.8+
2. 安裝依賴套件：

```bash
pip install -r requirements.txt
```

## 使用方法

```bash
python extract_pose.py [輸入路徑] [選項]
```

### 參數說明

- `input_path`: 輸入影片的檔案路徑，**或包含 mp4 檔案的資料夾路徑**（必填）
- `--output`, `-o`: 輸出檔案的路徑（若輸入為資料夾，則此參數為輸出資料夾路徑）。**支援 .json 或 .md 副檔名**。
- `--sample_rate`, `-s`: 採樣率，每 N 幀處理一次（預設：1，即處理每一幀）

### 範例

**單一檔案處理（輸出 JSON）：**
處理 `squat_standard.mp4` 並將結果保存為 `squat_data.json`：

```bash
python extract_pose.py squat_standard.mp4 -o squat_data.json
```

**批次處理資料夾：**
處理 `videos/` 資料夾內的所有 mp4 檔案，並將結果保存到 `results/` 資料夾（預設輸出為 .json）：

```bash
python extract_pose.py videos/ -o results/
```

## 輸出格式

### JSON 格式 (推薦)
包含完整的影片資訊與每一幀的關鍵點數據，結構如下：
```json
{
  "video_info": { ... },
  "frames": [
    {
      "frame_index": 0,
      "timestamp_sec": 0.0,
      "landmarks": [ ... ],       // 歸一化座標 (x, y, z)
      "world_landmarks": [ ... ]  // 真實世界 3D 座標 (x, y, z in meters)
    },
    ...
  ]
}
```

### Markdown 格式
若輸出檔名為 `.md`，則會產生包含摘要表格與 JSON 區塊的報告文件。
1. **影片資訊**：解析度、FPS、總幀數等。
2. **關鍵幀摘要**：以表格顯示重要關鍵點（鼻、肩、髖、膝、踝）的數值。
3. **完整數據 (JSON)**：包含所有 33 個關鍵點的完整 JSON 數據區塊。

## 關鍵點對照

MediaPipe Pose 提供 33 個關鍵點，本工具在摘要表格中重點顯示以下部位：
- 0: Nose (鼻子)
- 11: Left Shoulder (左肩)
- 12: Right Shoulder (右肩)
- 23: Left Hip (左髖)
- 24: Right Hip (右髖)
- 25: Left Knee (左膝)
- 26: Right Knee (右膝)
- 27: Left Ankle (左踝)
- 28: Right Ankle (右踝)
