import librosa
import soundfile as sf
import numpy as np
import os
import argparse # 導入 argparse 模組

def modify_audio_segment(
    input_audio_path: str,
    output_audio_path: str,
    start_sec: float,
    end_sec: float,
    pitch_semitones: float = 0.0,
    speed_factor: float = 1.0
):
    """
    修改音訊檔案中特定時間區間的音高和播放速度。

    :param input_audio_path: 輸入音訊檔案路徑。
    :param output_audio_path: 輸出修改後音訊檔案的路徑。
    :param start_sec: 開始修改的時間點（秒）。
    :param end_sec: 結束修改的時間點（秒）。
    :param pitch_semitones: 音高調整量（半音數）。正值提高音高，負值降低音高。
    :param speed_factor: 播放速度調整倍數。大於 1.0 加速，小於 1.0 減速。
    """
    if not os.path.exists(input_audio_path):
        print(f"錯誤：找不到輸入檔案 {input_audio_path}")
        return

    print(f"正在載入音訊：{input_audio_path}...")
    # 載入整個音訊檔案。y 是音訊數據，sr 是取樣率。
    y, sr = librosa.load(input_audio_path, sr=None, mono=True)
    total_duration = len(y) / sr
    
    # 處理 end_sec 預設值：如果 end_sec 未指定或設置為大於總時長，則使用總時長
    if end_sec is None or end_sec > total_duration:
        end_sec = total_duration

    if start_sec < 0 or start_sec >= end_sec or end_sec > total_duration:
        print(f"錯誤：時間範圍無效。總時長: {total_duration:.2f} 秒。提供的範圍: {start_sec:.2f}s - {end_sec:.2f}s。")
        return

    # 將時間 (秒) 轉換為取樣點索引 (Sample Indices)
    start_sample = librosa.time_to_samples(start_sec, sr=sr)
    end_sample = librosa.time_to_samples(end_sec, sr=sr)

    # 1. 切割音訊段
    pre_segment = y[:start_sample]
    target_segment = y[start_sample:end_sample]
    post_segment = y[end_sample:]

    print(f"目標修改時長：{end_sec - start_sec:.2f} 秒。")
    print(f"音高調整：{pitch_semitones} 半音，速度調整：{speed_factor} 倍。")

    modified_target = target_segment
    
    # --- 2. 應用速度調整 (Time Stretching) ---
    if speed_factor != 1.0:
        # time_stretch 函數的 rate 參數是調整的倍數。
        # rate > 1.0 -> 加速 (時長變短)
        # rate < 1.0 -> 減速 (時長變長)
        # Librosa 使用相位聲碼器 (Phase Vocoder) 進行時間伸縮，通常用於音樂。
        modified_target = librosa.effects.time_stretch(modified_target, rate=speed_factor)
        print(f"  -> 速度調整完成。新時長: {len(modified_target) / sr:.2f} 秒。")
    
    # --- 3. 應用音高調整 (Pitch Shifting) ---
    if pitch_semitones != 0.0:
        # pitch_shift 函數的 n_steps 參數是以半音為單位。
        modified_target = librosa.effects.pitch_shift(
            modified_target, 
            sr=sr, 
            n_steps=pitch_semitones
        )
        print(f"  -> 音高調整完成。")
        
    # --- 4. 重新拼接音訊 ---
    # 使用 numpy.concatenate 將三個 numpy 陣列重新組合成完整的音訊
    final_audio = np.concatenate([pre_segment, modified_target, post_segment])

    # --- 5. 儲存修改後的音訊檔案 ---
    # 使用 soundfile 儲存，以保留原始取樣率和位元深度。
    sf.write(output_audio_path, final_audio, sr)

    print(f"成功儲存修改後的音訊到：{output_audio_path} (總時長: {len(final_audio) / sr:.2f} 秒)")
    print("----------------------------------------")
# print(2)

# ====================================================================
# ARGPARSE 命令列參數處理
# ====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="修改音訊檔案中特定時間區間的音高和播放速度。",
        formatter_class=argparse.RawTextHelpFormatter # 保持描述格式
    )

    # 必填參數
    parser.add_argument(
        "--input_audio_path", 
        type=str, 
        help="輸入音訊檔案的路徑。"
    )
    parser.add_argument(
        "--output_audio_path", 
        type=str, 
        help="輸出修改後音訊檔案的路徑。"
    )
    
    # 選填參數
    parser.add_argument(
        "--start_sec", 
        type=float, 
        default=5.0, 
        help="開始修改的時間點 (秒)。預設為 0.0。"
    )
    parser.add_argument(
        "--end_sec", 
        type=float, 
        default=13.0, 
        help="結束修改的時間點 (秒)。預設為音訊總時長。"
    )
    parser.add_argument(
        "--pitch", 
        type=float, 
        default=10.0, 
        help="音高調整量 (半音數)。正值提高，負值降低。預設為 0.0。"
    )
    parser.add_argument(
        "--speed", 
        type=float, 
        default=1.5, 
        help="播放速度調整倍數。大於 1.0 加速，小於 1.0 減速。預設為 1.0。"
    )

    args = parser.parse_args()

    # 呼叫主修改函數
    modify_audio_segment(
        input_audio_path=args.input_audio_path,
        output_audio_path=args.output_audio_path,
        start_sec=args.start_sec,
        end_sec=args.end_sec,
        pitch_semitones=args.pitch,
        speed_factor=args.speed
    )


"""
python /Users/wtforbes/Library/CloudStorage/Dropbox/singing-helper/modify_audio.py \
    --input_audio_path /Users/wtforbes/Library/CloudStorage/Dropbox/singing-helper/關於小熊.flac \
    --output_audio_path /Users/wtforbes/Library/CloudStorage/Dropbox/singing-helper/關於小熊_mod.flac
"""