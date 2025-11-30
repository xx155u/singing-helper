import gradio as gr
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import soundfile as sf
import tempfile
import os
from librosa.sequence import dtw
import cv2
import threading
import time
from collections import deque
import subprocess
from librosa.onset import onset_detect, onset_strength

# 中文字體設定
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# --- 全域設定 (Global Settings) ---
TARGET_SR = 16000
HOP_LENGTH = 512


def get_audio_duration(audio_path):
    """獲取音訊檔案的時長（秒）。"""
    try:
        info = sf.info(audio_path)
        return info.duration
    except Exception as e:
        print(f"無法獲取音訊時長: {e}")
        return 30.0


def format_time(seconds):
    """將秒數格式化為易讀的時間字串"""
    if seconds >= 60:
        return f"{int(seconds // 60)} 分 {int(seconds % 60)} 秒"
    else:
        return f"{int(seconds)} 秒"


def extract_pseudo_onsets(features, smooth=5, threshold_ratio=0.3):
    """由 RMS 特徵偵測 pseudo-onsets（節奏強拍）。"""
    rms = features[:, -1]
    rms_smooth = np.convolve(rms, np.ones(smooth) / smooth, mode='same')
    rms_smooth = (rms_smooth - rms_smooth.min()) / (np.ptp(rms_smooth) + 1e-8)

    th = rms_smooth.mean() + threshold_ratio * rms_smooth.std()
    onsets = [i for i in range(1, len(rms_smooth))
              if rms_smooth[i] > th and rms_smooth[i] > rms_smooth[i-1]]
    return np.asarray(onsets, dtype=float)


def tempo_similarity(feat_in, feat_ref):
    """回傳 0–100 的節奏分數，越高越同步"""
    on1, on2 = extract_pseudo_onsets(feat_in), extract_pseudo_onsets(feat_ref)
    if len(on1) < 2 or len(on2) < 2:
        return 20.0

    L = max(len(on1), len(on2))
    a = np.pad(on1, (0, L-len(on1)), 'edge').reshape(-1, 1)
    b = np.pad(on2, (0, L-len(on2)), 'edge').reshape(-1, 1)

    D, _ = dtw(a, b)
    dist = D[-1, -1]
    score = 100 * np.exp(-dist / 50)
    return float(np.clip(score, 0, 100))


def generate_pitch_animation_video(pitch_timeline, mixed_audio_path, output_path=None):
    """使用純 matplotlib + ffmpeg 生成動畫影片"""
    if not pitch_timeline:
        print("⚠️ 無音高時間軸數據，跳過影片生成")
        return None
    
    if output_path is None:
        output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    
    print("📊 開始生成動畫影片...")
    
    times = np.array([d['time'] for d in pitch_timeline])
    pitch_diffs = np.array([d['pitch_diff'] for d in pitch_timeline])
    similarities = np.array([d['similarity'] for d in pitch_timeline])
    
    duration = times[-1]
    fps = 20
    total_frames = int(duration * fps)
    
    fig = plt.figure(figsize=(14, 10))
    
    def draw_frame(frame_idx):
        current_time = frame_idx / fps
        fig.clear()
        
        gs = fig.add_gridspec(3, 1, height_ratios=[1, 2, 2], hspace=0.3)
        
        ax_status = fig.add_subplot(gs[0])
        ax_status.axis('off')
        
        if len(times) > 0:
            idx = np.argmin(np.abs(times - current_time))
            current_data = pitch_timeline[idx]
            
            pitch_diff = current_data['pitch_diff']
            similarity = current_data['similarity']
            
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
        
        ax_pitch = fig.add_subplot(gs[1])
        window_size = 3.0
        x_min = max(0, current_time - window_size / 2)
        x_max = current_time + window_size / 2
        
        mask_played = times <= current_time
        mask_future = times > current_time
        
        colors_played = ['red' if abs(pd) > 1.5 else 'orange' if abs(pd) > 0.5 else 'green' 
                         for pd in pitch_diffs[mask_played]]
        
        if np.any(mask_played):
            ax_pitch.scatter(times[mask_played], pitch_diffs[mask_played], 
                            c=colors_played, alpha=0.8, s=50, zorder=3, edgecolors='black', linewidth=0.5)
            ax_pitch.plot(times[mask_played], pitch_diffs[mask_played], 
                         color='gray', alpha=0.5, linewidth=2, zorder=2)
        
        if np.any(mask_future):
            ax_pitch.scatter(times[mask_future], pitch_diffs[mask_future], 
                            c='lightgray', alpha=0.3, s=30, zorder=1)
        
        ax_pitch.axvline(x=current_time, color='blue', linestyle='-', linewidth=3, zorder=4)
        ax_pitch.axhspan(-0.5, 0.5, alpha=0.15, color='green', zorder=0)
        ax_pitch.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        
        ax_pitch.set_xlabel('時間 (秒)', fontsize=11)
        ax_pitch.set_ylabel('音高偏差 (半音)', fontsize=11)
        ax_pitch.set_title('即時音高偏差追蹤', fontsize=13, fontweight='bold')
        ax_pitch.grid(True, alpha=0.3)
        ax_pitch.set_ylim(-6, 6)
        ax_pitch.set_xlim(x_min, x_max)
        
        ax_similarity = fig.add_subplot(gs[2])
        
        if np.any(mask_played):
            ax_similarity.plot(times[mask_played], similarities[mask_played], 
                              color='#2E86AB', linewidth=3, zorder=3)
            ax_similarity.fill_between(times[mask_played], similarities[mask_played], 0,
                                      alpha=0.3, color='#2E86AB', zorder=2)
        
        if np.any(mask_future):
            ax_similarity.plot(times[mask_future], similarities[mask_future], 
                              color='lightgray', linewidth=2, alpha=0.5, zorder=1)
        
        ax_similarity.axvline(x=current_time, color='blue', linestyle='-', linewidth=3, zorder=4)
        ax_similarity.axhline(y=70, color='gray', linestyle='--', alpha=0.5)
        
        ax_similarity.set_xlabel('時間 (秒)', fontsize=11)
        ax_similarity.set_ylabel('相似度分數', fontsize=11)
        ax_similarity.set_title('即時音準相似度', fontsize=13, fontweight='bold')
        ax_similarity.grid(True, alpha=0.3)
        ax_similarity.set_ylim(0, 105)
        ax_similarity.set_xlim(x_min, x_max)

    temp_video = output_path.replace('.mp4', '_no_audio.mp4')
    
    try:
        print("⏳ 正在渲染動畫幀...")
        writer = animation.FFMpegWriter(fps=fps, bitrate=1800, codec='libx264')
        
        with writer.saving(fig, temp_video, dpi=100):
            for i in range(total_frames):
                draw_frame(i)
                writer.grab_frame()

        plt.close(fig)
        
        print("🎵 正在合併音訊...")

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
        
        if os.path.exists(temp_video):
            os.remove(temp_video)
        
        print(f"✅ 動畫影片生成完成！")
        return output_path
        
    except FileNotFoundError:
        print("❌ 錯誤：系統未安裝 ffmpeg")
        return None
    except Exception as e:
        print(f"❌ 生成影片時發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        if os.path.exists(temp_video):
            return temp_video
        return None


class PlaybackState:
    """播放狀態管理類"""
    def __init__(self):
        self.is_playing = False
        self.current_time = 0.0
        self.total_duration = 0.0
        self.pitch_data = []
        self.lock = threading.Lock()
        
playback_state = PlaybackState()


def precompute_pitch_differences(features_input, features_ref, interval=0.1):
    """預先計算整段音訊每個時間點的音高差異。"""
    frames_per_interval = int(interval * TARGET_SR / HOP_LENGTH)
    min_frames = min(features_input.shape[0], features_ref.shape[0])
    pitch_timeline = []
    
    for frame_idx in range(0, min_frames, frames_per_interval):
        window_size = int(0.5 * TARGET_SR / HOP_LENGTH)
        end_idx = min(frame_idx + window_size, min_frames)
        
        if end_idx - frame_idx < 10:
            continue
        
        chroma_input = features_input[frame_idx:end_idx, :12]
        chroma_ref = features_ref[frame_idx:end_idx, :12]
        
        input_pitch = np.argmax(np.mean(chroma_input, axis=0))
        ref_pitch = np.argmax(np.mean(chroma_ref, axis=0))
        
        pitch_diff = (input_pitch - ref_pitch) % 12
        if pitch_diff > 6:
            pitch_diff = pitch_diff - 12
        
        chroma_input_norm = chroma_input / (np.linalg.norm(chroma_input, axis=1, keepdims=True) + 1e-8)
        chroma_ref_norm = chroma_ref / (np.linalg.norm(chroma_ref, axis=1, keepdims=True) + 1e-8)
        similarity = np.mean(np.sum(chroma_input_norm * chroma_ref_norm, axis=1)) * 100
        
        timestamp = frame_idx * HOP_LENGTH / TARGET_SR
        
        pitch_timeline.append({
            'time': timestamp,
            'pitch_diff': pitch_diff,
            'similarity': similarity,
            'input_pitch': input_pitch,
            'ref_pitch': ref_pitch
        })
    
    return pitch_timeline


def get_note_name(pitch_class):
    """將音高類別轉換為音符名稱"""
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return note_names[int(pitch_class) % 12]


def windowed_pitch_analysis(features_input, features_ref, window_size=3.0, overlap=1.0):
    """使用滑動視窗分析音準，每個視窗獨立進行 DTW 和音準評估。"""
    frames_per_window = int(window_size * TARGET_SR / HOP_LENGTH)
    frames_per_step = int((window_size - overlap) * TARGET_SR / HOP_LENGTH)
    
    results = []
    min_frames = min(features_input.shape[0], features_ref.shape[0])
    
    window_start = 0
    window_idx = 0
    
    while window_start + frames_per_window <= min_frames:
        window_end = window_start + frames_per_window
        
        input_window = features_input[window_start:window_end]
        ref_window = features_ref[window_start:window_end]
        
        if input_window.shape[0] < 10 or ref_window.shape[0] < 10:
            window_start += frames_per_step
            continue
        
        try:
            D, wp = align_dtw(input_window, ref_window)
            
            chroma_input = input_window[:, :12]
            chroma_ref = ref_window[:, :12]
            
            pitch_score, pitch_deviation, pitch_direction = calculate_window_pitch_score(
                chroma_input, chroma_ref, D, wp
            )
            
            tempo_score = calculate_tempo_score(input_window, ref_window)
            overall_score = 0.7 * pitch_score + 0.3 * tempo_score
            
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
    """計算視窗內的音準分數和偏差。"""
    input_peak_bins = np.argmax(chroma_input, axis=1)
    ref_peak_bins = np.argmax(chroma_ref, axis=1)
    
    pitch_diff = np.mean(input_peak_bins) - np.mean(ref_peak_bins)
    
    chroma_input_norm = chroma_input / (np.linalg.norm(chroma_input, axis=1, keepdims=True) + 1e-8)
    chroma_ref_norm = chroma_ref / (np.linalg.norm(chroma_ref, axis=1, keepdims=True) + 1e-8)
    
    cosine_similarities = np.sum(chroma_input_norm * chroma_ref_norm, axis=1)
    avg_similarity = np.mean(cosine_similarities)
    
    pitch_score = max(0, min(100, avg_similarity * 100))
    
    if abs(pitch_diff) < 0.5:
        pitch_direction = "準確"
    elif pitch_diff > 0:
        pitch_direction = "偏高"
    else:
        pitch_direction = "偏低"
    
    return pitch_score, abs(pitch_diff), pitch_direction


def calculate_tempo_score(input_feat, ref_feat):
    """節奏分數 (0–100)。"""
    return tempo_similarity(input_feat, ref_feat)


def plot_windowed_analysis(results):
    """繪製視窗分析結果的時間序列圖。"""
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
    
    ax2 = axes[1]
    colors = ['red' if r['pitch_direction'] == '偏高' else 'blue' if r['pitch_direction'] == '偏低' else 'green' 
              for r in results]
    
    ax2.bar(times, pitch_deviations, width=2, color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='可接受偏差 (0.5半音)')
    
    ax2.set_xlabel('時間 (秒)', fontsize=12)
    ax2.set_ylabel('音高偏差 (半音)', fontsize=12)
    ax2.set_title('各時段音高偏差分析', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
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
    """根據視窗分析結果生成文字建議。"""
    if not results:
        return "音訊長度不足，無法進行視窗分析。"
    
    feedback = []
    
    avg_pitch_score = np.mean([r['pitch_score'] for r in results])
    avg_tempo_score = np.mean([r['tempo_score'] for r in results])
    avg_overall = np.mean([r['overall_score'] for r in results])
    
    feedback.append(f"### 📊 整體表現統計")
    feedback.append(f"- **平均音準分數**: {avg_pitch_score:.1f} 分")
    feedback.append(f"- **平均節奏分數**: {avg_tempo_score:.1f} 分")
    feedback.append(f"- **平均整體分數**: {avg_overall:.1f} 分")
    feedback.append("")
    
    problem_windows = [r for r in results if r['overall_score'] < 70]
    
    if problem_windows:
        feedback.append(f"### ⚠️ 需要改進的時段（共 {len(problem_windows)} 個）")
        for i, window in enumerate(problem_windows[:5], 1):
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
    
    best_window = max(results, key=lambda r: r['overall_score'])
    feedback.append(f"### 🌟 最佳時段")
    feedback.append(f"**{best_window['time_start']:.1f}秒 - {best_window['time_end']:.1f}秒** "
                   f"(整體分數: {best_window['overall_score']:.1f})")
    
    return '\n'.join(feedback)


def extract_features(audio_path):
    """載入音訊、標準化處理，並提取 Chroma 和 RMS 特徵。"""
    if not audio_path or not os.path.exists(audio_path):
        raise gr.Error("請錄製或上傳有效的音訊檔案。")

    try:
        y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    except Exception as e:
        raise gr.Error(f"載入音訊檔案失敗: {e}")

    MIN_SAMPLES = 2048
    if len(y) < MIN_SAMPLES:
        raise gr.Error(f"音訊長度過短 ({len(y)/sr:.2f} 秒)，無法進行有效分析。")

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
    features = np.vstack([chroma, rms])
    
    return features.T


def align_dtw(features_input, features_ref):
    """執行動態時間規整 (DTW) 以對齊兩段音訊的時間軸。"""
    D, wp = librosa.sequence.dtw(
        X=features_input.T,
        Y=features_ref.T,
        metric='euclidean'
    )
    return D, wp


def mix_audio(path1, path2):
    """將兩段音訊混合成一個檔案以便於比較。"""
    try:
        y1, _ = librosa.load(path1, sr=TARGET_SR, mono=True)
        y2, _ = librosa.load(path2, sr=TARGET_SR, mono=True)

        len1, len2 = len(y1), len(y2)
        if len1 > len2:
            y2 = np.pad(y2, (0, len1 - len2))
        else:
            y1 = np.pad(y1, (0, len2 - len1))

        mixed = y1 + y2
        
        max_amp = np.max(np.abs(mixed))
        if max_amp > 0:
            mixed /= max_amp
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
            sf.write(fp.name, mixed, TARGET_SR)
            return fp.name
    except Exception as e:
        print(f"混合音訊時發生錯誤: {e}")
        return None


# ============================================================
# 【修改】主要評估函數 - 移除進度條，保留資訊卡片
# ============================================================
def singing_evaluator(input_audio_path, ref_audio_path, generate_video=True):
    """
    Gradio 介面的主要處理函數。
    不使用進度條，直接處理並返回結果。
    """
    if not input_audio_path or not ref_audio_path:
        raise gr.Error("請同時上傳或錄製您的歌聲和參考音訊。")

    try:
        # 1. 特徵提取
        features_input = extract_features(input_audio_path)
        features_ref = extract_features(ref_audio_path)
        
        # 2. 視窗式分析
        window_results = windowed_pitch_analysis(features_input, features_ref, 
                                                 window_size=3.0, overlap=1.0)
        
        # 3. 計算整體分數
        if window_results:
            avg_score = np.mean([r['overall_score'] for r in window_results])
            similarity_score = f"{avg_score:.1f}"
        else:
            similarity_score = "N/A"
        
        # 4. 生成文字建議
        feedback_text = generate_windowed_feedback(window_results)
        
        # 5. 可視化視窗分析結果
        windowed_plot = plot_windowed_analysis(window_results)

        # 6. 混合音訊
        mixed_audio_path = mix_audio(input_audio_path, ref_audio_path)
        
        # 7. 預計算即時音高數據
        pitch_timeline = precompute_pitch_differences(features_input, features_ref, interval=0.1)
        
        # 8. 生成動畫影片（可選）
        animation_video_path = None
        if generate_video and pitch_timeline:
            try:
                animation_video_path = generate_pitch_animation_video(
                    pitch_timeline, mixed_audio_path
                )
            except Exception as e:
                print(f"❌ 生成動畫影片失敗: {e}")
                animation_video_path = None

        return (similarity_score, feedback_text, windowed_plot, 
                input_audio_path, ref_audio_path, mixed_audio_path,
                animation_video_path)
        
    except gr.Error as e:
        raise e
    except Exception as e:
        error_message = f"分析過程中發生未知錯誤: {e}"
        print(error_message)
        import traceback
        traceback.print_exc()
        raise gr.Error("分析失敗，請檢查您的音訊檔案是否有效，或稍後再試。")


# ============================================================
# 【新增】生成預估時間資訊卡片 HTML
# ============================================================
def generate_info_card_html(input_audio_path, ref_audio_path, generate_video):
    """生成顯示預估時間的資訊卡片 HTML"""
    input_duration = get_audio_duration(input_audio_path)
    ref_duration = get_audio_duration(ref_audio_path)
    max_duration = max(input_duration, ref_duration)
    
    # 預估處理時間
    if generate_video:
        estimated_time = max_duration * 2.5
    else:
        estimated_time = max_duration * 1.5
    estimated_time = max(estimated_time, 10.0)
    
    est_str = format_time(estimated_time)
    duration_str = format_time(max_duration)
    
    html = f"""
<div style="padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; color: white; font-family: 'Microsoft YaHei', sans-serif; margin-bottom: 15px;">
    <div style="text-align: center; margin-bottom: 15px;">
        <span style="font-size: 20px; font-weight: bold;">⏳ 正在分析中...</span>
    </div>
    
    <div style="display: flex; justify-content: space-around; text-align: center;">
        <div>
            <div style="font-size: 14px; opacity: 0.9;">📊 音訊長度</div>
            <div style="font-size: 18px; font-weight: bold; margin-top: 5px;">{duration_str}</div>
        </div>
        <div>
            <div style="font-size: 14px; opacity: 0.9;">⏱️ 預估處理時間</div>
            <div style="font-size: 18px; font-weight: bold; margin-top: 5px;">{est_str}</div>
        </div>
    </div>
    
    <div style="margin-top: 15px; text-align: center; font-size: 13px; opacity: 0.8;">
        {"🎬 包含動畫影片生成" if generate_video else "📊 僅進行音訊分析"}
    </div>
</div>
"""
    return html


def generate_complete_card_html(elapsed_time):
    """生成完成狀態的資訊卡片 HTML"""
    elapsed_str = format_time(elapsed_time)
    
    html = f"""
<div style="padding: 20px; background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); border-radius: 12px; color: white; font-family: 'Microsoft YaHei', sans-serif; margin-bottom: 15px;">
    <div style="text-align: center;">
        <span style="font-size: 24px;">✅ 分析完成！</span>
    </div>
    <div style="margin-top: 10px; text-align: center; font-size: 14px;">
        總耗時: {elapsed_str}
    </div>
</div>
"""
    return html


# --- Gradio 界面定義 ---
title = "🎙️ AI 歌聲相似性評估與輔助系統 🎶"
description = (
    "上傳或**即時錄製**您的歌聲和參考音訊，系統將使用**滑動視窗分析**（每3秒一個視窗，重疊1秒），"
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
    
    # 【新增】資訊卡片顯示區
    info_card_display = gr.HTML(value="", visible=False)
    
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
        start_time = time.time()
        
        (score, feedback, plot, _, _, mixed_path, 
        animation_path) = singing_evaluator(input_audio_path, ref_audio_path, 
                                            generate_video=should_generate_video)
        
        elapsed_time = time.time() - start_time
        complete_html = generate_complete_card_html(elapsed_time)
        
        return {
            info_card_display: gr.HTML(value=complete_html, visible=True),
            result_outputs_group: gr.Column(visible=True),
            score_display: score,
            feedback_output: feedback,
            windowed_plot_output: plot,
            input_playback: gr.Audio(value=input_audio_path, label="您的歌聲 (Input)"),
            ref_playback: gr.Audio(value=ref_audio_path, label="參考音訊 (Reference)"),
            mixed_playback: gr.Audio(value=mixed_path, label="🎛️ 疊加播放 (Mixed for Comparison)"),
            animation_video_output: gr.Video(value=animation_path, label="🎵 即時音高分析動畫")
        }
    
    def prepare_ui(input_audio_path, ref_audio_path, should_generate_video):
        """
        階段 1: 點擊後立即執行
        - 顯示預估時間資訊卡片
        - 隱藏舊結果
        - 鎖定按鈕
        """
        if not input_audio_path or not ref_audio_path:
            raise gr.Error("請同時上傳或錄製您的歌聲和參考音訊。")
        
        info_html = generate_info_card_html(input_audio_path, ref_audio_path, should_generate_video)
        
        return (
            gr.HTML(value=info_html, visible=True),
            gr.Column(visible=False),
            gr.Button(value="⏳ 評估中請稍後...", interactive=False)
        )

    def finish_analysis():
        """階段 3: 分析完成後執行 - 恢復按鈕狀態"""
        return gr.Button(value="🚀 開始分析與評估", interactive=True)
    
    # 事件綁定
    analyze_btn.click(
        fn=prepare_ui, 
        inputs=[input_audio, ref_audio, generate_video_checkbox], 
        outputs=[info_card_display, result_outputs_group, analyze_btn],
        queue=False
    ).then(
        fn=run_analysis,
        inputs=[input_audio, ref_audio, generate_video_checkbox],
        outputs=[info_card_display, result_outputs_group, score_display, feedback_output, windowed_plot_output, 
                 input_playback, ref_playback, mixed_playback, animation_video_output]
    ).then(
        fn=finish_analysis,
        inputs=[],
        outputs=[analyze_btn]
    )


if __name__ == "__main__":
    demo.launch(share=True)