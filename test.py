import gradio as gr
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import io

# 核心邏輯 1: 特徵提取 (Feature Extraction)
def extract_features(audio_path, sr=22050):
    """載入音訊並提取 Chroma (音高/和聲) 和 RMS (能量/音量) 特徵。"""
    try:
        # 載入音訊
        y, sr = librosa.load(audio_path, sr=sr, mono=True)
    except Exception as e:
        # Gradio 在錄音時，如果使用者沒有錄製內容就提交，可能會傳入 None 或空路徑
        if not audio_path:
            raise gr.Error("請錄製或上傳音訊檔案。")
        raise gr.Error(f"載入音訊檔案失敗: {e}")

    # --- 新增 ---
    # 檢查音訊長度，避免因檔案過短導致 librosa 分析時出錯
    MIN_SAMPLES = 2048 # FFT 運算需要一定的樣本數
    if len(y) < MIN_SAMPLES:
        raise gr.Error(f"音訊長度過短 ({len(y)/sr:.2f} 秒)，無法進行有效分析。請提供至少 {MIN_SAMPLES/sr:.2f} 秒的音訊。")

    # 1. Chroma feature (音高/和聲內容)
    # 使用 CQT (Constant-Q Transform) 得到更好的音高解析度，再轉換為 Chroma
    # Chroma 向量 (12維) 代表了每個八度內的音高分佈
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    
    # 2. RMS (能量/音量) - 作為簡單的歌聲活動偵測 (VAD)
    rms = librosa.feature.rms(y=y)
    
    # 合併特徵：[12維 Chroma, 1維 RMS]
    # 轉置後成為 (N_frames, 13) 的矩陣
    features = np.vstack([chroma, rms])
    
    # 返回轉置後的特徵矩陣和取樣率
    return features.T, sr, y

# 核心邏輯 2: DTW 對齊 (DTW Alignment)
def align_dtw(features_input, features_ref):
    """
    執行動態時間規整 (DTW) 以對齊兩段歌聲的時間軸。
    DTW 尋找從 (0, 0) 到 (N, M) 成本最低的路徑，以最小化時間規整後的差異。
    """
    # 使用歐幾里得距離作為距離度量 (metric)
    D, wp = librosa.sequence.dtw(
        X=features_input.T,  # X 軸 (Input)
        Y=features_ref.T,    # Y 軸 (Reference)
        metric='euclidean'
    )
    # --- 修正 ---
    # librosa.dtw 返回的 wp (warping path) 是一個 (N_path, 2) 的陣列，
    # 後續分析程式碼 `for k, (i, j) in enumerate(wp):` 預期的是這種形狀。
    # 原本的 `wp.T` 會將其轉置為 (2, N_path)，導致迴圈解包時發生 "too many values to unpack" 錯誤。
    # 因此，我們直接返回原始的 wp。
    return D, wp

# 核心邏輯 3: 結果分析與建議生成 (Analysis and Feedback Generation)
def analyze_results(wp, D, features_input, features_ref, sr):
    """根據 DTW 結果分析速度和音高差異，並生成建議。"""
    feedback = []
    
    # 計算每幀的時間長度
    HOP_LENGTH = 512 # Librosa 預設的 Frame Hop 長度
    TIME_PER_FRAME = HOP_LENGTH / sr
    
    # --- 1. 整體相似度分數 (Overall Similarity Score) ---
    # 總成本 (Total Cost) 
    total_cost = D[-1, -1]
    # 正規化成本 (Normalized Cost) 作為相似度指標 (值越低越相似)
    normalized_cost = total_cost / len(wp)
    similarity_score = f"{normalized_cost:.4f}"
    
    # --- 2. 整體節奏/速度分析 (Global Tempo Analysis) ---
    input_frames = features_input.shape[0]
    ref_frames = features_ref.shape[0]
    avg_slope = input_frames / ref_frames
    
    tempo_suggestion = "**整體節奏評估 (Global Tempo):** "
    if avg_slope > 1.15: # Input 比 Reference 長 15% 以上 (唱得太慢)
        tempo_suggestion += f"您的歌聲比參考音訊慢了約 {avg_slope * 100 - 100:.1f}%。建議您整體加快演唱速度。"
    elif avg_slope < 0.85: # Input 比 Reference 短 15% 以上 (唱得太快)
        tempo_suggestion += f"您的歌聲比參考音訊快了約 {100 - avg_slope * 100:.1f}%。建議您整體放慢演唱速度。"
    else:
        tempo_suggestion += "您的整體節奏掌握得很好，與參考音訊大致同步。"
    
    feedback.append(tempo_suggestion)
    
    # --- 3. 局部時序與音高問題 (Local Timing and Pitch Issues) ---
    feedback.append("\n**具體局部改進建議 (Local Feedback):**")
    
    # 計算每條路徑點的局部成本 (Local Cost) 作為不匹配程度的指標
    local_costs = np.zeros(len(wp))
    for k, (i, j) in enumerate(wp):
        # 使用 cost matrix D 的值，並除以路徑距離 (i+j) 來進行局部比較
        local_costs[k] = D[i, j] / (i + j + 1e-6) # 加上一個極小值避免除以零

    # 找出局部成本顯著高於平均值 (例如 1.5 個標準差以上) 的點
    mean_cost = np.mean(local_costs)
    std_cost = np.std(local_costs)
    threshold = mean_cost + 1.5 * std_cost 
    
    high_cost_points = np.where(local_costs > threshold)[0]
    
    if high_cost_points.size == 0:
        feedback.append("• 沒有檢測到明顯的局部時序或音高問題，表現優異！")
    else:
        # 將連續的高成本點分組為單個問題區段 (Issue Segment)
        issue_groups = []
        if high_cost_points.size > 0:
            current_group = [high_cost_points[0]]
            for i in range(1, len(high_cost_points)):
                if high_cost_points[i] == high_cost_points[i-1] + 1:
                    current_group.append(high_cost_points[i])
                else:
                    issue_groups.append(current_group)
                    current_group = [high_cost_points[i]]
            issue_groups.append(current_group)
        
        # 限制最多顯示 5 個建議，避免版面過長
        for group in issue_groups[:5]:
            start_k, end_k = group[0], group[-1]
            
            # 對應到 Input 和 Reference 的幀索引
            i_start, j_start = wp[start_k]
            i_end, j_end = wp[end_k]
            
            # 將 Input 幀索引轉換為時間 (秒)
            time_start = i_start * TIME_PER_FRAME
            time_end = i_end * TIME_PER_FRAME
            
            # --- 局部音高 (Pitch) 分析 ---
            # Chroma 特徵位於 features_input/ref 的前 12 行 (0-11)
            chroma_input_seg = features_input[i_start:i_end+1, :12]
            chroma_ref_seg = features_ref[j_start:j_end+1, :12]
            
            pitch_suggestion = ""
            if chroma_input_seg.size > 0 and chroma_ref_seg.size > 0:
                # 計算該區段內 Input 和 Ref 的主要音高 (Peak Chroma Bin)
                input_peak_bin = np.mean(np.argmax(chroma_input_seg, axis=1))
                ref_peak_bin = np.mean(np.argmax(chroma_ref_seg, axis=1))
                # pitch_diff 正值: 偏高 (Input Chroma Bin > Ref Chroma Bin)
                pitch_diff = input_peak_bin - ref_peak_bin 
                
                # 計算局部速度 (Local Speed) 差異
                frames_input_seg = i_end - i_start
                frames_ref_seg = j_end - j_start
                local_slope = frames_input_seg / (frames_ref_seg + 1e-6) # 避免除以零

                if abs(pitch_diff) > 0.8: # 音高差異超過約 0.8 個半音
                    pitch_level = "偏高" if pitch_diff > 0 else "偏低"
                    pitch_suggestion = f"音高明顯**{pitch_level}**，平均偏差約 {abs(pitch_diff):.1f} 個半音。"
                elif local_slope > 1.8:
                    pitch_suggestion = "**時序嚴重拖沓**，您的演唱太慢了，需要更果斷地進入下一個樂句。"
                elif local_slope < 0.5:
                    pitch_suggestion = "**時序嚴重超前**，您的演唱太快了，請仔細聆聽參考音訊的間隔。"
                else:
                    pitch_suggestion = "音準或音色不匹配。請專注於該樂句的音準穩定性。"

            
                feedback.append(f"• **時間 {time_start:.2f} 秒到 {time_end:.2f} 秒:** {pitch_suggestion}")
        
        if len(issue_groups) > 5:
            feedback.append("• ... (僅顯示前 5 個最顯著的問題點)")

    return similarity_score, '\n'.join(feedback)

# 核心邏輯 4: DTW 可視化 (DTW Visualization)
def plot_dtw_path(D, wp, sr):
    """繪製累積成本矩陣和最佳規整路徑。"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    
    # 顯示累積成本矩陣 (Accumulated Cost Matrix)
    # D 的形狀為 (input_frames, ref_frames)。specshow 將第一維對應 y 軸，第二維對應 x 軸。
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='time', ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f', label='累積成本 (Accumulated Cost)')
    ax.set(title='DTW 累積成本矩陣與最佳路徑')
    
    # 繪製最佳規整路徑 (Warping Path)
    # wp[:, 0] 是 input frames, wp[:, 1] 是 reference frames.
    # 我們繪製 (ref_time, input_time) 來對應 x, y 軸。
    ax.plot(librosa.frames_to_time(wp[:, 1], sr=sr), librosa.frames_to_time(wp[:, 0], sr=sr), 
            marker='o', color='red', linestyle='-', linewidth=2, alpha=0.5, 
            label='最佳規整路徑 (Warping Path)')
    
    ax.set_xlabel("參考音訊時間 (Reference Time)")
    ax.set_ylabel("您的歌聲時間 (Input Time)")
    ax.legend(loc='lower right')
    
    # --- 修正 ---
    # Gradio 的 gr.Plot 元件可以直接處理 matplotlib 的 Figure 物件，
    # 這比手動轉換為 bytes 更簡潔。
    # 我們不應在此處呼叫 plt.close(fig)，否則 Gradio 將無法渲染它。
    return fig


# Gradio 主要函數
def singing_evaluator(input_audio_path, ref_audio_path):
    """Gradio 接口的主要處理函數。"""
    if not input_audio_path or not ref_audio_path:
        raise gr.Error("請上傳或錄製您的歌聲和參考音訊檔案。")

    try:
        # 1. 特徵提取
        features_input, sr, y_input = extract_features(input_audio_path)
        features_ref, sr, y_ref = extract_features(ref_audio_path)
        
        # 2. DTW 對齊
        D, wp = align_dtw(features_input, features_ref)
        
        # 3. 分析並生成建議
        similarity_score, feedback_text = analyze_results(wp, D, features_input, features_ref, sr)
        
        # 4. 可視化 DTW 路徑
        dtw_plot = plot_dtw_path(D, wp, sr)
        
        # 5. 返回結果
        return similarity_score, feedback_text, dtw_plot, input_audio_path, ref_audio_path
        
    except gr.Error as e:
        # 直接拋出 Gradio 的錯誤，使其能清晰地顯示在 UI 上
        raise e
    except Exception as e:
        error_message = f"分析過程中發生未知錯誤: {e}"
        print(error_message) # 在後端打印詳細錯誤以供調試
        # 向用戶顯示一個更友好的錯誤訊息
        raise gr.Error("分析失敗，請檢查您的音訊檔案是否有效，或稍後再試。")

# --- Gradio 界面定義 ---

# 描述和標題
title = "🎙️ AI 歌聲相似性評估與輔助系統 (支援錄音) 🎶"
description = (
    "上傳或**即時錄製**兩段音訊檔案（您的歌聲和參考音訊），系統將使用動態時間規整 (DTW) 技術對齊音軌，"
    "分析節奏和音高差異，並提供具體的改進建議，例如幾秒到幾秒太快或音高偏低。"
)

with gr.Blocks(theme=gr.themes.Soft(), title=title) as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(description)

    # 輸入區 (已加入 microphone 來源)
    with gr.Row():
        input_audio_upload = gr.Audio(
            type="filepath", 
            label="🎤 您的歌聲 (Input Audio)", 
            sources=["upload", "microphone"] # 支援上傳和錄音
        )
        ref_audio_upload = gr.Audio(
            type="filepath", 
            label="🎧 參考音訊 (Reference Audio)", 
            sources=["upload", "microphone"] # 支援上傳和錄音
        )
    
    analyze_btn = gr.Button("🚀 開始分析與評估", variant="primary")
    
    # 輸出區
    # 將 gr.Column 賦值給一個變數，以便在 .success() 中引用
    result_outputs_group = gr.Column(visible=False) 
    with result_outputs_group:
        gr.Markdown("---")
        gr.Markdown("## 📋 評估結果")
        
        with gr.Row():
            score_display = gr.Textbox(label="總體正規化相似度分數 (越低越相似)", show_label=True, scale=1)
            
        # 將 Markdown 元件也賦值給一個變數
        feedback_output = gr.Markdown("### 📜 具體改進建議")

        dtw_plot_output = gr.Plot(label="DTW 規整路徑圖 (Warping Path Visualization)")
        
        gr.Markdown("---")
        gr.Markdown("## 🔊 同步播放 (請手動按下播放鍵)")
        
        with gr.Row():
            aligned_input_playback = gr.Audio(label="您的歌聲 (Input)", interactive=False, autoplay=False)
            aligned_ref_playback = gr.Audio(label="參考音訊 (Reference)", interactive=False, autoplay=False)
        gr.Markdown("*注意：Gradio 僅提供並排播放，您需要手動同時點擊播放以進行聽覺上的同步比對。DTW 結果是演算法上已對齊的。*")

    # 綁定事件
    # 點擊按鈕後，先將結果區域設為隱藏，以清除舊的結果
    def hide_results():
        return gr.Column(visible=False)

    analyze_btn.click(
        fn=hide_results,
        inputs=None,
        outputs=[result_outputs_group]
    ).then(
        fn=singing_evaluator,
        inputs=[input_audio_upload, ref_audio_upload],
        outputs=[score_display, feedback_output, dtw_plot_output, aligned_input_playback, aligned_ref_playback]
    ).success(
        fn=lambda: gr.Column(visible=True), # 成功後再顯示結果
        inputs=None,
        outputs=result_outputs_group
    )

if __name__ == "__main__":
    demo.launch(share=True)
