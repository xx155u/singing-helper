import gradio as gr
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import soundfile as sf
import tempfile
import os
import cv2
import threading
import time
from collections import deque
import subprocess

# 中文字體設定
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 全域設定
TARGET_SR = 16000
HOP_LENGTH = 512

def generate_pitch_animation_video(pitch_timeline, mixed_audio_path, output_path=None, progress=gr.Progress()):
    """
    使用純 matplotlib + ffmpeg 生成動畫影片
    相容新版 matplotlib，移除所有 emoji 符號
    """
    if not pitch_timeline:
        print("⚠️ 無音高時間軸數據，跳過影片生成")
        return None
    
    if output_path is None:
        output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    
    progress(0, desc="準備生成動畫影片...")
    print("📊 開始生成動畫影片...")
    
    # 提取數據
    times = np.array([d['time'] for d in pitch_timeline])
    pitch_diffs = np.array([d['pitch_diff'] for d in pitch_timeline])
    similarities = np.array([d['similarity'] for d in pitch_timeline])
    
    duration = times[-1]
    fps = 20
    
    progress(0.1, desc="創建動畫圖形...")
    # 創建圖形
    fig = plt.figure(figsize=(14, 10))
    
    def animate(frame):
        """動畫更新函數（移除所有 emoji）"""
        current_time = frame / fps
        fig.clear()
        
        gs = fig.add_gridspec(3, 1, height_ratios=[1, 2, 2], hspace=0.3)
        
        # === 頂部：即時狀態顯示 ===
        ax_status = fig.add_subplot(gs[0])
        ax_status.axis('off')
        
        if len(times) > 0:
            idx = np.argmin(np.abs(times - current_time))
            current_data = pitch_timeline[idx]
            
            pitch_diff = current_data['pitch_diff']
            similarity = current_data['similarity']
            
            # 狀態判斷（改用純文字標籤）
            if abs(pitch_diff) < 0.5:
                status = "[準確] 音準準確"
                status_color = 'green'
            elif abs(pitch_diff) < 1.5:
                status = "[注意] 輕微偏差"
                status_color = 'orange'
            else:
                status = "[警告] 明顯偏差"
                status_color = 'red'
            
            direction = "偏高" if pitch_diff > 0 else "偏低" if pitch_diff < 0 else "準確"
            
            # 顯示狀態（使用矩形背景突顯）
            bbox_props = dict(boxstyle='round,pad=0.5', facecolor=status_color, alpha=0.2)
            status_text = f"{status}\n當前時間: {current_time:.2f} 秒"
            ax_status.text(0.5, 0.7, status_text, 
                          fontsize=18, fontweight='bold', 
                          ha='center', va='center', 
                          color=status_color, bbox=bbox_props)
            
            detail_text = (f"音高偏差: {abs(pitch_diff):.2f} 半音 ({direction}) | "
                          f"相似度: {similarity:.1f} 分")
            ax_status.text(0.5, 0.3, detail_text,
                          fontsize=11, ha='center', va='center', color='black')
        
        # === 音高偏差追蹤圖 ===
        ax_pitch = fig.add_subplot(gs[1])
        window_size = 10.0
        x_min = max(0, current_time - window_size / 2)
        x_max = current_time + window_size / 2
        
        mask_played = times <= current_time
        mask_future = times > current_time
        
        colors_played = ['red' if abs(pd) > 1.5 else 'orange' if abs(pd) > 0.5 else 'green' 
                         for pd in pitch_diffs[mask_played]]
        
        # 繪製已播放部分（高亮）
        if np.any(mask_played):
            ax_pitch.scatter(times[mask_played], pitch_diffs[mask_played], 
                            c=colors_played, alpha=0.8, s=50, zorder=3, edgecolors='black', linewidth=0.5)
            ax_pitch.plot(times[mask_played], pitch_diffs[mask_played], 
                         color='gray', alpha=0.5, linewidth=2, zorder=2)
        
        # 繪製未播放部分（灰色）
        if np.any(mask_future):
            ax_pitch.scatter(times[mask_future], pitch_diffs[mask_future], 
                            c='lightgray', alpha=0.3, s=30, zorder=1)
        
        # 當前播放位置標記
        ax_pitch.axvline(x=current_time, color='blue', linestyle='-', linewidth=3, zorder=4)
        
        # 標記安全區域
        ax_pitch.axhspan(-0.5, 0.5, alpha=0.15, color='green', zorder=0)
        ax_pitch.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        
        ax_pitch.set_xlabel('時間 (秒)', fontsize=11)
        ax_pitch.set_ylabel('音高偏差 (半音)', fontsize=11)
        ax_pitch.set_title('即時音高偏差追蹤', fontsize=13, fontweight='bold')
        ax_pitch.grid(True, alpha=0.3)
        ax_pitch.set_ylim(-6, 6)
        ax_pitch.set_xlim(x_min, x_max)
        
        # === 相似度追蹤圖 ===
        ax_similarity = fig.add_subplot(gs[2])
        
        # 繪製已播放部分
        if np.any(mask_played):
            ax_similarity.plot(times[mask_played], similarities[mask_played], 
                              color='#2E86AB', linewidth=3, zorder=3)
            ax_similarity.fill_between(times[mask_played], similarities[mask_played], 0,
                                      alpha=0.3, color='#2E86AB', zorder=2)
        
        # 繪製未播放部分
        if np.any(mask_future):
            ax_similarity.plot(times[mask_future], similarities[mask_future], 
                              color='lightgray', linewidth=2, alpha=0.5, zorder=1)
        
        # 當前播放位置標記
        ax_similarity.axvline(x=current_time, color='blue', linestyle='-', linewidth=3, zorder=4)
        ax_similarity.axhline(y=70, color='gray', linestyle='--', alpha=0.5)
        ax_similarity.fill_between([x_min, x_max], 70, 100, alpha=0.1, color='green')
        ax_similarity.fill_between([x_min, x_max], 0, 70, alpha=0.1, color='red')
        
        ax_similarity.set_xlabel('時間 (秒)', fontsize=11)
        ax_similarity.set_ylabel('相似度分數', fontsize=11)
        ax_similarity.set_title('即時音準相似度', fontsize=13, fontweight='bold')
        ax_similarity.grid(True, alpha=0.3)
        ax_similarity.set_ylim(0, 105)
        ax_similarity.set_xlim(x_min, x_max)
    
    # 生成動畫
    total_frames = int(duration * fps)
    anim = animation.FuncAnimation(fig, animate, frames=total_frames, 
                                   interval=1000/fps, blit=False)
    
    # 保存無音訊的影片
    temp_video = output_path.replace('.mp4', '_no_audio.mp4')
    
    try:
        progress(0.3, desc="正在渲染動畫幀...")
        print("⏳ 正在渲染動畫幀（這可能需要一些時間）...")
        writer = animation.FFMpegWriter(fps=fps, bitrate=1800, codec='libx264')
        anim.save(temp_video, writer=writer, dpi=100)
        plt.close(fig)
        
        progress(0.8, desc="正在合併音訊...")
        print("🎵 正在合併音訊...")
        # 使用 ffmpeg 合併音訊
        result = subprocess.run([
            'ffmpeg', '-y', '-loglevel', 'error',
            '-i', temp_video,
            '-i', mixed_audio_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-shortest',
            output_path
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"⚠️ ffmpeg 警告: {result.stderr}")
        
        # 清理臨時檔案
        if os.path.exists(temp_video):
            os.remove(temp_video)
        
        progress(1.0, desc="動畫影片生成完成！")
        print(f"✅ 動畫影片生成完成！")
        return output_path
        
    except FileNotFoundError:
        print("❌ 錯誤：系統未安裝 ffmpeg")
        print("請執行：brew install ffmpeg")
        return None
    except Exception as e:
        print(f"❌ 生成影片時發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        # 如果合併失敗，返回無音訊版本
        if os.path.exists(temp_video):
            print(f"⚠️ 返回無音訊版本: {temp_video}")
            return temp_video
        return None

# --- 即時播放分析全域變數 ---
class PlaybackState:
    """播放狀態管理類"""
    def __init__(self):
        self.is_playing = False
        self.current_time = 0.0
        self.total_duration = 0.0
        self.pitch_data = []
        self.lock = threading.Lock()
        
playback_state = PlaybackState()

# 核心邏輯 7: 預計算音高差異數據 (Pre-compute Pitch Difference Data)
def precompute_pitch_differences(features_input, features_ref, interval=0.1, progress=gr.Progress()):
    """
    預先計算整段音訊每個時間點的音高差異。
    """
    progress(0, desc="開始預計算音高數據...")
    
    frames_per_interval = int(interval * TARGET_SR / HOP_LENGTH)
    min_frames = min(features_input.shape[0], features_ref.shape[0])
    
    pitch_timeline = []
    total_intervals = max(1, min_frames // frames_per_interval)
    
    for idx, frame_idx in enumerate(progress.tqdm(range(0, min_frames, frames_per_interval), 
                                                   desc="計算音高差異")):
        # 取一小段特徵進行分析（使用 0.5 秒的視窗）
        window_size = int(0.5 * TARGET_SR / HOP_LENGTH)
        end_idx = min(frame_idx + window_size, min_frames)
        
        if end_idx - frame_idx < 10:  # 視窗太小就跳過
            continue
        
        # 提取 Chroma 特徵
        chroma_input = features_input[frame_idx:end_idx, :12]
        chroma_ref = features_ref[frame_idx:end_idx, :12]
        
        # 計算平均音高
        input_pitch = np.argmax(np.mean(chroma_input, axis=0))
        ref_pitch = np.argmax(np.mean(chroma_ref, axis=0))
        
        # 計算音高偏差（半音）
        pitch_diff = (input_pitch - ref_pitch) % 12
        if pitch_diff > 6:
            pitch_diff = pitch_diff - 12
        
        # 計算音準相似度
        chroma_input_norm = chroma_input / (np.linalg.norm(chroma_input, axis=1, keepdims=True) + 1e-8)
        chroma_ref_norm = chroma_ref / (np.linalg.norm(chroma_ref, axis=1, keepdims=True) + 1e-8)
        similarity = np.mean(np.sum(chroma_input_norm * chroma_ref_norm, axis=1)) * 100
        
        # 計算時間戳
        timestamp = frame_idx * HOP_LENGTH / TARGET_SR
        
        pitch_timeline.append({
            'time': timestamp,
            'pitch_diff': pitch_diff,
            'similarity': similarity,
            'input_pitch': input_pitch,
            'ref_pitch': ref_pitch
        })
    
    return pitch_timeline


# 核心邏輯 8: 即時播放控制與視覺化更新 (Real-time Playback Control)
def update_realtime_display(pitch_timeline, current_time):
    """
    根據當前播放時間更新即時音高顯示。
    """
    if not pitch_timeline:
        return "暫無數據", None
    
    # 找到當前時間最接近的數據點
    closest_data = min(pitch_timeline, key=lambda x: abs(x['time'] - current_time))
    
    # 格式化顯示文本
    pitch_diff = closest_data['pitch_diff']
    similarity = closest_data['similarity']
    
    if abs(pitch_diff) < 0.5:
        status = "✅ 音準準確"
        color = "🟢"
    elif abs(pitch_diff) < 1.5:
        status = "⚠️ 輕微偏差"
        color = "🟡"
    else:
        status = "❌ 明顯偏差"
        color = "🔴"
    
    direction = "偏高" if pitch_diff > 0 else "偏低" if pitch_diff < 0 else "準確"
    
    display_text = f"""
### {color} 即時音高分析 ({current_time:.2f} 秒)

**狀態**: {status}  
**音高偏差**: {abs(pitch_diff):.2f} 半音 ({direction})  
**音準相似度**: {similarity:.1f} 分  

---
**音符對照**:
- 您的音高: {get_note_name(closest_data['input_pitch'])}
- 參考音高: {get_note_name(closest_data['ref_pitch'])}
"""
    
    # 生成即時圖表
    plot_fig = plot_realtime_pitch(pitch_timeline, current_time)
    
    return display_text, plot_fig


def get_note_name(pitch_class):
    """將音高類別轉換為音符名稱"""
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return note_names[int(pitch_class) % 12]


def plot_realtime_pitch(pitch_timeline, current_time, window_size=10.0):
    """
    繪製即時音高差異圖表，顯示當前播放位置。
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    if not pitch_timeline:
        ax1.text(0.5, 0.5, '暫無數據', ha='center', va='center', fontsize=14)
        ax2.text(0.5, 0.5, '暫無數據', ha='center', va='center', fontsize=14)
        plt.tight_layout()
        return fig
    
    times = [d['time'] for d in pitch_timeline]
    pitch_diffs = [d['pitch_diff'] for d in pitch_timeline]
    similarities = [d['similarity'] for d in pitch_timeline]
    
    # 圖表 1: 音高偏差
    colors = ['red' if abs(pd) > 1.5 else 'orange' if abs(pd) > 0.5 else 'green' 
              for pd in pitch_diffs]
    
    ax1.scatter(times, pitch_diffs, c=colors, alpha=0.6, s=30)
    ax1.plot(times, pitch_diffs, color='gray', alpha=0.3, linewidth=1)
    
    # 標記當前播放位置
    ax1.axvline(x=current_time, color='blue', linestyle='--', linewidth=2, label='當前播放位置')
    
    # 標記安全區域
    ax1.axhspan(-0.5, 0.5, alpha=0.2, color='green', label='準確範圍')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    
    ax1.set_xlabel('時間 (秒)', fontsize=11)
    ax1.set_ylabel('音高偏差 (半音)', fontsize=11)
    ax1.set_title('即時音高偏差追蹤', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-6, 6)
    
    # 設定 x 軸範圍（顯示當前位置前後的視窗）
    x_min = max(0, current_time - window_size / 2)
    x_max = current_time + window_size / 2
    ax1.set_xlim(x_min, x_max)
    
    # 圖表 2: 音準相似度
    ax2.plot(times, similarities, color='#2E86AB', linewidth=2, label='音準相似度')
    ax2.fill_between(times, 70, 100, alpha=0.1, color='green')
    ax2.fill_between(times, 0, 70, alpha=0.1, color='red')
    
    # 標記當前播放位置
    ax2.axvline(x=current_time, color='blue', linestyle='--', linewidth=2, label='當前播放位置')
    ax2.axhline(y=70, color='gray', linestyle='--', alpha=0.5, label='良好門檻')
    
    ax2.set_xlabel('時間 (秒)', fontsize=11)
    ax2.set_ylabel('相似度分數 (0-100)', fontsize=11)
    ax2.set_title('即時音準相似度', fontsize=13, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)
    ax2.set_xlim(x_min, x_max)
    
    plt.tight_layout()
    return fig

# 核心邏輯 5: 視窗式音準分析 (Windowed Pitch Analysis)
def windowed_pitch_analysis(features_input, features_ref, window_size=5.0, overlap=2.0, progress=gr.Progress()):
    """
    使用滑動視窗分析音準，每個視窗獨立進行 DTW 和音準評估。
    """
    progress(0, desc="開始視窗式分析...")
    
    # 計算每個視窗的幀數
    frames_per_window = int(window_size * TARGET_SR / HOP_LENGTH)
    frames_per_step = int((window_size - overlap) * TARGET_SR / HOP_LENGTH)
    
    results = []
    
    total_input_frames = features_input.shape[0]
    total_ref_frames = features_ref.shape[0]
    
    # 使用較短的音訊長度作為基準
    min_frames = min(total_input_frames, total_ref_frames)
    
    window_start = 0
    window_idx = 0
    
    # 計算總視窗數
    total_windows = max(1, (min_frames - frames_per_window) // frames_per_step + 1)
    
    while window_start + frames_per_window <= min_frames:
        progress(window_idx / total_windows, desc=f"分析視窗 {window_idx+1}/{total_windows}")
        
        window_end = window_start + frames_per_window
        
        # 提取當前視窗的特徵
        input_window = features_input[window_start:window_end]
        ref_window = features_ref[window_start:window_end]
        
        # 檢查視窗大小
        if input_window.shape[0] < 10 or ref_window.shape[0] < 10:
            window_start += frames_per_step
            continue
        
        try:
            # 對視窗進行 DTW 對齊
            D, wp = align_dtw(input_window, ref_window)
            
            # 提取 Chroma 特徵（前 12 維）
            chroma_input = input_window[:, :12]
            chroma_ref = ref_window[:, :12]
            
            # 計算音準分數和偏差
            pitch_score, pitch_deviation, pitch_direction = calculate_window_pitch_score(
                chroma_input, chroma_ref, D, wp
            )
            
            # 計算節奏分數
            tempo_score = calculate_tempo_score(len(input_window), len(ref_window))
            
            # 整體視窗分數
            overall_score = 0.7 * pitch_score + 0.3 * tempo_score
            
            # 記錄時間資訊
            time_start = window_start * HOP_LENGTH / TARGET_SR
            time_end = window_end * HOP_LENGTH / TARGET_SR
            
            results.append({
                'window_idx': window_idx,
                'time_start': time_start,
                'time_end': time_end,
                'time_center': (time_start + time_end) / 2,
                'pitch_score': pitch_score,
                'pitch_deviation': pitch_deviation,
                'pitch_direction': pitch_direction,
                'tempo_score': tempo_score,
                'overall_score': overall_score,
                'normalized_dtw_cost': D[-1, -1] / len(wp)
            })
            
        except Exception as e:
            print(f"視窗 {window_idx} 分析失敗: {e}")
        
        window_start += frames_per_step
        window_idx += 1
    
    return results


def calculate_window_pitch_score(chroma_input, chroma_ref, D, wp):
    """
    計算視窗內的音準分數和偏差。
    """
    # 計算平均音高
    input_peak_bins = np.argmax(chroma_input, axis=1)
    ref_peak_bins = np.argmax(chroma_ref, axis=1)
    
    # 計算音高偏差
    pitch_diff = np.mean(input_peak_bins) - np.mean(ref_peak_bins)
    
    # 計算音準相似度（使用餘弦相似度）
    chroma_input_norm = chroma_input / (np.linalg.norm(chroma_input, axis=1, keepdims=True) + 1e-8)
    chroma_ref_norm = chroma_ref / (np.linalg.norm(chroma_ref, axis=1, keepdims=True) + 1e-8)
    
    cosine_similarities = np.sum(chroma_input_norm * chroma_ref_norm, axis=1)
    avg_similarity = np.mean(cosine_similarities)
    
    # 轉換為分數 (0-100)
    pitch_score = max(0, min(100, avg_similarity * 100))
    
    # 判斷偏差方向
    if abs(pitch_diff) < 0.5:
        pitch_direction = "準確"
    elif pitch_diff > 0:
        pitch_direction = "偏高"
    else:
        pitch_direction = "偏低"
    
    return pitch_score, abs(pitch_diff), pitch_direction


def calculate_tempo_score(input_frames, ref_frames):
    """計算節奏分數"""
    tempo_ratio = input_frames / (ref_frames + 1e-8)
    
    # 理想比例為 1，偏離越多分數越低
    tempo_deviation = abs(tempo_ratio - 1.0)
    tempo_score = max(0, 100 * (1 - tempo_deviation * 2))
    
    return tempo_score


# 核心邏輯 6: 視窗分析結果可視化 (Windowed Analysis Visualization)
def plot_windowed_analysis(results):
    """
    繪製視窗分析結果的時間序列圖。
    """
    if not results:
        fig = plt.figure(figsize=(12, 6))
        plt.text(0.5, 0.5, '無足夠資料進行視窗分析', 
                ha='center', va='center', fontsize=14)
        return fig
    
    times = [r['time_center'] for r in results]
    pitch_scores = [r['pitch_score'] for r in results]
    tempo_scores = [r['tempo_score'] for r in results]
    overall_scores = [r['overall_score'] for r in results]
    pitch_deviations = [r['pitch_deviation'] for r in results]
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 第一張圖：分數隨時間變化
    ax1 = axes[0]
    ax1.plot(times, pitch_scores, marker='o', label='音準分數', linewidth=2, color='#2E86AB')
    ax1.plot(times, tempo_scores, marker='s', label='節奏分數', linewidth=2, color='#A23B72')
    ax1.plot(times, overall_scores, marker='^', label='整體分數', linewidth=2.5, color='#F18F01')
    
    ax1.axhline(y=70, color='gray', linestyle='--', alpha=0.5, label='良好門檻 (70分)')
    ax1.fill_between(times, 70, 100, alpha=0.1, color='green')
    ax1.fill_between(times, 0, 70, alpha=0.1, color='red')
    
    ax1.set_xlabel('時間 (秒)', fontsize=12)
    ax1.set_ylabel('分數 (0-100)', fontsize=12)
    ax1.set_title('各時段音準與節奏評估', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)
    
    # 第二張圖：音高偏差
    ax2 = axes[1]
    colors = ['red' if r['pitch_direction'] == '偏高' else 'blue' if r['pitch_direction'] == '偏低' else 'green' 
              for r in results]
    
    bars = ax2.bar(times, pitch_deviations, width=2, color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='可接受偏差 (0.5半音)')
    
    ax2.set_xlabel('時間 (秒)', fontsize=12)
    ax2.set_ylabel('音高偏差 (半音)', fontsize=12)
    ax2.set_title('各時段音高偏差分析', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加顏色圖例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label='偏高'),
        Patch(facecolor='blue', alpha=0.7, label='偏低'),
        Patch(facecolor='green', alpha=0.7, label='準確')
    ]
    ax2.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    return fig


def generate_windowed_feedback(results):
    """
    根據視窗分析結果生成文字建議。
    """
    if not results:
        return "音訊長度不足，無法進行視窗分析。"
    
    feedback = []
    
    # 整體統計
    avg_pitch_score = np.mean([r['pitch_score'] for r in results])
    avg_tempo_score = np.mean([r['tempo_score'] for r in results])
    avg_overall = np.mean([r['overall_score'] for r in results])
    
    feedback.append(f"### 📊 整體表現統計")
    feedback.append(f"- **平均音準分數**: {avg_pitch_score:.1f} 分")
    feedback.append(f"- **平均節奏分數**: {avg_tempo_score:.1f} 分")
    feedback.append(f"- **平均整體分數**: {avg_overall:.1f} 分")
    feedback.append("")
    
    # 找出問題時段（分數低於 70 的）
    problem_windows = [r for r in results if r['overall_score'] < 70]
    
    if problem_windows:
        feedback.append(f"### ⚠️ 需要改進的時段（共 {len(problem_windows)} 個）")
        for i, window in enumerate(problem_windows[:5], 1):  # 最多顯示 5 個
            time_range = f"{window['time_start']:.1f}秒 - {window['time_end']:.1f}秒"
            
            issues = []
            if window['pitch_score'] < 70:
                issues.append(f"音準{window['pitch_direction']}（偏差 {window['pitch_deviation']:.1f} 半音）")
            if window['tempo_score'] < 70:
                issues.append("節奏不穩")
            
            feedback.append(f"{i}. **{time_range}**: {', '.join(issues)}")
        
        if len(problem_windows) > 5:
            feedback.append(f"   ... 還有 {len(problem_windows) - 5} 個時段需要注意")
    else:
        feedback.append("### ✅ 表現優異")
        feedback.append("所有時段的表現都在良好水準以上，請繼續保持！")
    
    feedback.append("")
    
    # 最佳時段
    best_window = max(results, key=lambda r: r['overall_score'])
    feedback.append(f"### 🌟 最佳時段")
    feedback.append(f"**{best_window['time_start']:.1f}秒 - {best_window['time_end']:.1f}秒** "
                   f"(整體分數: {best_window['overall_score']:.1f})")
    
    return '\n'.join(feedback)


# 核心邏輯 1: 特徵提取 (Feature Extraction)
def extract_features(audio_path, progress=gr.Progress()):
    """載入音訊、標準化處理，並提取 Chroma 和 RMS 特徵。"""
    if not audio_path or not os.path.exists(audio_path):
        raise gr.Error("請錄製或上傳有效的音訊檔案。")

    try:
        progress(0, desc="載入音訊檔案...")
        # 載入音訊，並重取樣至目標 SR，轉換為單聲道
        y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    except Exception as e:
        raise gr.Error(f"載入音訊檔案失敗: {e}")

    # 檢查音訊長度是否足夠進行分析
    MIN_SAMPLES = 2048
    if len(y) < MIN_SAMPLES:
        raise gr.Error(f"音訊長度過短 ({len(y)/sr:.2f} 秒)，無法進行有效分析。")

    progress(0.3, desc="提取 Chroma 特徵...")
    # 1. Chroma feature (音高/和聲內容)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)
    
    progress(0.7, desc="提取 RMS 特徵...")
    # 2. RMS (Root-Mean-Square Energy for volume)
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
    
    # 合併特徵並轉置 -> (N_frames, 13)
    features = np.vstack([chroma, rms])
    
    progress(1.0, desc="特徵提取完成")
    return features.T

# 核心邏輯 2: DTW 對齊 (DTW Alignment)
def align_dtw(features_input, features_ref):
    """
    執行動態時間規整 (DTW) 以對齊兩段音訊的時間軸。
    """
    # 使用歐幾里得距離作為成本度量
    D, wp = librosa.sequence.dtw(
        X=features_input.T,
        Y=features_ref.T,
        metric='euclidean'
    )
    return D, wp

# 核心邏輯 3: 結果分析與建議生成 (Analysis and Feedback Generation)
def analyze_results(wp, D, features_input, features_ref):
    """根據 DTW 結果分析速度和音高差異，並生成建議。"""
    feedback = []
    
    # --- 1. 整體相似度分數 (Overall Similarity Score) ---
    # 正規化 DTW 成本 (值越低越相似)
    normalized_cost = D[-1, -1] / len(wp)
    
    # 將成本轉換為 0-100 的分數，越高越好
    k = 2.0 
    similarity_score = 100 * np.exp(-k * normalized_cost)
    
    # 偵測是否為不同歌曲
    DIFFERENT_SONG_THRESHOLD = 1.0 
    if normalized_cost > DIFFERENT_SONG_THRESHOLD:
        feedback.append(
            "**分析警示：** 兩段音訊的差異過大，系統判斷可能來自**不同的歌曲**。"
            "因此相似度分數極低，以下的局部建議可能不具參考價值。"
        )
        return f"{similarity_score:.1f}", '\n'.join(feedback)

    # --- 2. 整體節奏/速度分析 (Global Tempo Analysis) ---
    input_frames = features_input.shape[0]
    ref_frames = features_ref.shape[0]
    avg_slope = input_frames / ref_frames
    
    tempo_suggestion = "**整體節奏評估:** "
    if avg_slope > 1.15:
        tempo_suggestion += f"您的演唱速度比參考音訊慢了約 {avg_slope * 100 - 100:.0f}%，建議整體加快。"
    elif avg_slope < 0.85:
        tempo_suggestion += f"您的演唱速度比參考音訊快了約 {100 - avg_slope * 100:.0f}%，建議整體放慢。"
    else:
        tempo_suggestion += "您的整體節奏掌握得很好，與參考音訊大致同步。"
    feedback.append(tempo_suggestion)
    
    # --- 3. 局部時序與音高問題 (Local Timing and Pitch Issues) ---
    feedback.append("\n**具體局部改進建議:**")
    
    # 計算路徑上每個點的局部成本
    local_costs = np.array([D[i, j] for i, j in wp])
    # 計算每一步的成本增量，更能反應問題點
    step_costs = np.diff(local_costs, prepend=0)

    # 找出成本增量顯著高於平均值的點
    mean_step_cost = np.mean(step_costs)
    std_step_cost = np.std(step_costs)
    threshold = mean_step_cost + 1.5 * std_step_cost
    
    high_cost_indices = np.where(step_costs > threshold)[0]
    
    if high_cost_indices.size == 0:
        feedback.append("• 表現優異！未檢測到明顯的局部時序或音高問題。")
    else:
        # 將連續的高成本點分組為問題區段
        issue_groups = []
        if high_cost_indices.size > 0:
            current_group = [high_cost_indices[0]]
            for i in range(1, len(high_cost_indices)):
                # 如果索引是連續的，則視為同一問題
                if high_cost_indices[i] == high_cost_indices[i-1] + 1:
                    current_group.append(high_cost_indices[i])
                else:
                    issue_groups.append(current_group)
                    current_group = [high_cost_indices[i]]
            issue_groups.append(current_group)
        
        # 分析每個問題區段
        TIME_PER_FRAME = HOP_LENGTH / TARGET_SR
        for group in issue_groups[:5]: # 最多顯示前 5 個問題
            start_k, end_k = group[0], group[-1]
            i_start, j_start = wp[start_k]
            i_end, j_end = wp[end_k]
            
            time_start = i_start * TIME_PER_FRAME
            time_end = i_end * TIME_PER_FRAME
            
            # 提取該區段的 Chroma 特徵
            chroma_input_seg = features_input[i_start:i_end+1, :12]
            chroma_ref_seg = features_ref[j_start:j_end+1, :12]
            
            suggestion = ""
            if chroma_input_seg.size > 0 and chroma_ref_seg.size > 0:
                # 計算局部速度差異
                frames_input_seg = i_end - i_start
                frames_ref_seg = j_end - j_start
                local_slope = frames_input_seg / (frames_ref_seg + 1e-6)

                # 計算音高差異
                input_peak_bin = np.mean(np.argmax(chroma_input_seg, axis=1))
                ref_peak_bin = np.mean(np.argmax(chroma_ref_seg, axis=1))
                pitch_diff = input_peak_bin - ref_peak_bin
                
                if abs(pitch_diff) > 0.8: # 音高差異顯著
                    pitch_level = "偏高" if pitch_diff > 0 else "偏低"
                    suggestion = f"音高明顯**{pitch_level}** (偏差約 {abs(pitch_diff):.1f} 個半音)。"
                elif local_slope > 1.8:
                    suggestion = "**節奏拖沓**，演唱速度過慢。"
                elif local_slope < 0.5:
                    suggestion = "**節奏搶拍**，演唱速度過快。"
                else:
                    suggestion = "音準或音色不匹配，請注意此處的發聲穩定性。"
            
            if time_end - time_start < TIME_PER_FRAME:
                feedback.append(f"• **在 {time_start:.2f} 秒附近:** {suggestion}")
            else:
                feedback.append(f"• **時間 {time_start:.2f} 秒 - {time_end:.2f} 秒:** {suggestion}")
        
        if len(issue_groups) > 5:
            feedback.append("• ... (僅顯示前 5 個最顯著的問題點)")

    return f"{similarity_score:.1f}", '\n'.join(feedback)

# 核心邏輯 4: DTW 可視化 (DTW Visualization)
def plot_dtw_path(D, wp):
    """繪製累積成本矩陣和最佳規整路徑。"""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    
    img = librosa.display.specshow(D, sr=TARGET_SR, x_axis='frames', y_axis='frames', ax=ax, hop_length=HOP_LENGTH)
    fig.colorbar(img, ax=ax, label='累積成本')
    ax.set(title='DTW 累積成本與最佳路徑')
    
    # 繪製最佳路徑
    ax.plot(wp[:, 1], wp[:, 0], marker='.', color='red', linestyle='-', linewidth=2, alpha=0.6)
    
    ax.set_xlabel("參考音訊 (影格)")
    ax.set_ylabel("您的歌聲 (影格)")
    plt.tight_layout()
    return fig

# 混合音訊 (Mix Audio)
def mix_audio(path1, path2):
    """將兩段音訊混合成一個檔案以便於比較。"""
    try:
        y1, _ = librosa.load(path1, sr=TARGET_SR, mono=True)
        y2, _ = librosa.load(path2, sr=TARGET_SR, mono=True)

        # 填充較短的音訊
        len1, len2 = len(y1), len(y2)
        if len1 > len2:
            y2 = np.pad(y2, (0, len1 - len2))
        else:
            y1 = np.pad(y1, (0, len2 - len1))

        mixed = y1 + y2
        
        # 正規化以防止削波
        max_amp = np.max(np.abs(mixed))
        if max_amp > 0:
            mixed /= max_amp
        
        # 儲存到暫存檔
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
            sf.write(fp.name, mixed, TARGET_SR)
            return fp.name
    except Exception as e:
        print(f"混合音訊時發生錯誤: {e}")
        return None

def singing_evaluator(input_audio_path, ref_audio_path, generate_video=True, progress=gr.Progress()):
    """Gradio 介面的主要處理函數 - 使用視窗分析。"""
    if not input_audio_path or not ref_audio_path:
        raise gr.Error("請同時上傳或錄製您的歌聲和參考音訊。")

    try:
        # 1. 特徵提取
        progress(0, desc="開始分析...")
        progress(0.05, desc="提取輸入音訊特徵...")
        features_input = extract_features(input_audio_path, progress)
        
        progress(0.15, desc="提取參考音訊特徵...")
        features_ref = extract_features(ref_audio_path, progress)
        
        # 2. 視窗式分析（5秒視窗，2秒重疊）
        progress(0.25, desc="執行視窗式音準分析...")
        window_results = windowed_pitch_analysis(features_input, features_ref, 
                                                 window_size=5.0, overlap=2.0, progress=progress)
        
        # 3. 計算整體分數
        progress(0.55, desc="計算整體分數...")
        if window_results:
            avg_score = np.mean([r['overall_score'] for r in window_results])
            similarity_score = f"{avg_score:.1f}"
        else:
            similarity_score = "N/A"
        
        # 4. 生成文字建議
        progress(0.6, desc="生成改進建議...")
        feedback_text = generate_windowed_feedback(window_results)
        
        # 5. 可視化視窗分析結果
        progress(0.65, desc="生成視覺化圖表...")
        windowed_plot = plot_windowed_analysis(window_results)

        # 6. 混合音訊
        progress(0.7, desc="混合音訊...")
        mixed_audio_path = mix_audio(input_audio_path, ref_audio_path)
        
        # 7. 預計算即時音高數據
        progress(0.75, desc="預計算即時音高數據...")
        pitch_timeline = precompute_pitch_differences(features_input, features_ref, interval=0.1, progress=progress)
        
        # 8. 生成動畫影片（可選）
        animation_video_path = None
        if generate_video and pitch_timeline:
            try:
                progress(0.85, desc="生成動畫影片...")
                print("🎬 開始生成動畫影片...")
                animation_video_path = generate_pitch_animation_video(
                    pitch_timeline, mixed_audio_path, progress=progress
                )
            except Exception as e:
                print(f"❌ 生成動畫影片失敗: {e}")
                animation_video_path = None
        
        # 9. 返回所有結果
        progress(1.0, desc="分析完成！")
        return (similarity_score, feedback_text, windowed_plot, 
                input_audio_path, ref_audio_path, mixed_audio_path,
                animation_video_path)
        
    except gr.Error as e:
        raise e
    except Exception as e:
        error_message = f"分析過程中發生未知錯誤: {e}"
        print(error_message)
        raise gr.Error("分析失敗，請檢查您的音訊檔案是否有效，或稍後再試。")
    
# --- Gradio 界面定義 ---
title = "🎙️ AI 歌聲相似性評估與輔助系統 🎶"
description = (
    "上傳或**即時錄製**您的歌聲和參考音訊，系統將使用**滑動視窗分析**（每5秒一個視窗，重疊2秒），"
    "透過動態時間規整 (DTW) 技術，對每個時段的**音準、節奏**進行獨立評估。"
    "提供 **0-100 的整體分數**、**時間序列分析圖表**，以及**動態影片式即時音高分析**。"
)

with gr.Blocks(theme=gr.themes.Soft(), title=title) as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(description)

    with gr.Row():
        input_audio = gr.Audio(type="filepath", label="🎤 您的歌聲 (Input)", sources=["upload", "microphone"])
        ref_audio = gr.Audio(type="filepath", label="🎧 參考音訊 (Reference)", sources=["upload", "microphone"])
    
    with gr.Row():
        analyze_btn = gr.Button("🚀 開始分析與評估", variant="primary", scale=3)
        generate_video_checkbox = gr.Checkbox(label="生成動畫影片（需要較長時間）", value=True, scale=1)
    
    result_outputs_group = gr.Column(visible=False) 
    with result_outputs_group:
        gr.Markdown("---")
        gr.Markdown("## 📋 評估報告")
        
        with gr.Row():
            score_display = gr.Textbox(label="總體相似度分數 (0-100分，越高越好)", scale=1)
        
        feedback_output = gr.Markdown(label="### 📜 具體改進建議")

        windowed_plot_output = gr.Plot(label="📊 視窗式分析圖表 (音準與節奏隨時間變化)")
        
        gr.Markdown("---")
        gr.Markdown("## 🔊 音訊比對播放")
        
        with gr.Row():
            input_playback = gr.Audio(label="您的歌聲 (Input)")
            ref_playback = gr.Audio(label="參考音訊 (Reference)")
        
        mixed_playback = gr.Audio(label="🎛️ 疊加播放 (Mixed for Comparison)")
        gr.Markdown("*提示：播放「疊加播放」音軌，可以幫助您更清晰地聽出節奏和音準的差異。*")
        
        gr.Markdown("---")
        gr.Markdown("## 🎬 動態即時音高分析影片")
        gr.Markdown("*影片會自動與音訊同步播放，展示每個時刻的音高變化和偏差分析*")
        
        animation_video_output = gr.Video(label="🎵 即時音高分析動畫", autoplay=False)
        gr.Markdown("💡 **使用提示**: 點擊播放按鈕，影片會同步顯示音高分析動畫，讓您清楚看到每個時間點的表現")

    # 主分析流程
    def run_analysis(input_audio_path, ref_audio_path, should_generate_video):
        (score, feedback, plot, _, _, mixed_path, 
         animation_path) = singing_evaluator(input_audio_path, ref_audio_path, 
                                            generate_video=should_generate_video)
        
        return {
            result_outputs_group: gr.Column(visible=True),
            score_display: score,
            feedback_output: feedback,
            windowed_plot_output: plot,
            input_playback: gr.Audio(value=input_audio_path, label="您的歌聲 (Input)"),
            ref_playback: gr.Audio(value=ref_audio_path, label="參考音訊 (Reference)"),
            mixed_playback: gr.Audio(value=mixed_path, label="🎛️ 疊加播放 (Mixed for Comparison)"),
            animation_video_output: gr.Video(value=animation_path, label="🎵 即時音高分析動畫")
        }

    # 事件綁定
    analyze_btn.click(
        fn=lambda: gr.Column(visible=False),
        outputs=[result_outputs_group]
    ).then(
        fn=run_analysis,
        inputs=[input_audio, ref_audio, generate_video_checkbox],
        outputs=[result_outputs_group, score_display, feedback_output, windowed_plot_output, 
                input_playback, ref_playback, mixed_playback, animation_video_output]
    )

if __name__ == "__main__":
    demo.launch(share=True)