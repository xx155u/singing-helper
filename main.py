# import gradio as gr
# import numpy as np
# import librosa
# import librosa.display
# import matplotlib.pyplot as plt
# import os
# import io

# # 核心邏輯 1: 特徵提取 (Feature Extraction)
# def extract_features(audio_path, sr=22050):
#     """載入音訊並提取 Chroma (音高/和聲) 和 RMS (能量/音量) 特徵。"""
#     try:
#         # 載入音訊
#         y, sr = librosa.load(audio_path, sr=sr, mono=True)
#     except Exception as e:
#         # Gradio 在錄音時，如果使用者沒有錄製內容就提交，可能會傳入 None 或空路徑
#         if not audio_path:
#             raise gr.Error("請錄製或上傳音訊檔案。")
#         raise gr.Error(f"載入音訊檔案失敗: {e}")

#     # --- 新增 ---
#     # 檢查音訊長度，避免因檔案過短導致 librosa 分析時出錯
#     MIN_SAMPLES = 2048 # FFT 運算需要一定的樣本數
#     if len(y) < MIN_SAMPLES:
#         raise gr.Error(f"音訊長度過短 ({len(y)/sr:.2f} 秒)，無法進行有效分析。請提供至少 {MIN_SAMPLES/sr:.2f} 秒的音訊。")

#     # 1. Chroma feature (音高/和聲內容)
#     # 使用 CQT (Constant-Q Transform) 得到更好的音高解析度，再轉換為 Chroma
#     # Chroma 向量 (12維) 代表了每個八度內的音高分佈
#     chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    
#     # 2. RMS (能量/音量) - 作為簡單的歌聲活動偵測 (VAD)
#     rms = librosa.feature.rms(y=y)
    
#     # 合併特徵：[12維 Chroma, 1維 RMS]
#     # 轉置後成為 (N_frames, 13) 的矩陣
#     features = np.vstack([chroma, rms])
    
#     # 返回轉置後的特徵矩陣和取樣率
#     return features.T, sr, y

# # 核心邏輯 2: DTW 對齊 (DTW Alignment)
# def align_dtw(features_input, features_ref):
#     """
#     執行動態時間規整 (DTW) 以對齊兩段歌聲的時間軸。
#     DTW 尋找從 (0, 0) 到 (N, M) 成本最低的路徑，以最小化時間規整後的差異。
#     """
#     # 使用歐幾里得距離作為距離度量 (metric)
#     D, wp = librosa.sequence.dtw(
#         X=features_input.T,  # X 軸 (Input)
#         Y=features_ref.T,    # Y 軸 (Reference)
#         metric='euclidean'
#     )

#     return D, wp

# # 核心邏輯 3: 結果分析與建議生成 (Analysis and Feedback Generation)
# def analyze_results(wp, D, features_input, features_ref, sr):
#     """根據 DTW 結果分析速度和音高差異，並生成建議。"""
#     feedback = []
    
#     # 計算每幀的時間長度
#     HOP_LENGTH = 512 # Librosa 預設的 Frame Hop 長度
#     TIME_PER_FRAME = HOP_LENGTH / sr
    
#     # --- 1. 整體相似度分數 (Overall Similarity Score) ---
#     # 總成本 (Total Cost) 
#     total_cost = D[-1, -1]

#     # TODO:  I need score normalized to 0~100
#     # 正規化成本 (Normalized Cost) 作為相似度指標 (值越低越相似)
#     normalized_cost = total_cost / len(wp)
#     similarity_score = f"{normalized_cost:.4f}"
    
#     # --- 2. 整體節奏/速度分析 (Global Tempo Analysis) ---
#     input_frames = features_input.shape[0]
#     ref_frames = features_ref.shape[0]
#     avg_slope = input_frames / ref_frames
    
#     tempo_suggestion = "**整體節奏評估 (Global Tempo):** "
#     if avg_slope > 1.15: # Input 比 Reference 長 15% 以上 (唱得太慢)
#         tempo_suggestion += f"您的歌聲比參考音訊慢了約 {avg_slope * 100 - 100:.1f}%。建議您整體加快演唱速度。"
#     elif avg_slope < 0.85: # Input 比 Reference 短 15% 以上 (唱得太快)
#         tempo_suggestion += f"您的歌聲比參考音訊快了約 {100 - avg_slope * 100:.1f}%。建議您整體放慢演唱速度。"
#     else:
#         tempo_suggestion += "您的整體節奏掌握得很好，與參考音訊大致同步。"
    
#     feedback.append(tempo_suggestion)
    
#     # --- 3. 局部時序與音高問題 (Local Timing and Pitch Issues) ---
#     feedback.append("\n**具體局部改進建議 (Local Feedback):**")
    
#     # 計算每條路徑點的局部成本 (Local Cost) 作為不匹配程度的指標
#     local_costs = np.zeros(len(wp))
#     for k, (i, j) in enumerate(wp):
#         # 使用 cost matrix D 的值，並除以路徑距離 (i+j) 來進行局部比較
#         local_costs[k] = D[i, j] / (i + j + 1e-6) # 加上一個極小值避免除以零

#     # 找出局部成本顯著高於平均值 (例如 1.5 個標準差以上) 的點
#     mean_cost = np.mean(local_costs)
#     std_cost = np.std(local_costs)
#     threshold = mean_cost + 1.5 * std_cost 
    
#     high_cost_points = np.where(local_costs > threshold)[0]
    
#     if high_cost_points.size == 0:
#         feedback.append("• 沒有檢測到明顯的局部時序或音高問題，表現優異！")
#     else:
#         # 將連續的高成本點分組為單個問題區段 (Issue Segment)
#         issue_groups = []
#         if high_cost_points.size > 0:
#             current_group = [high_cost_points[0]]
#             for i in range(1, len(high_cost_points)):
#                 if high_cost_points[i] == high_cost_points[i-1] + 1:
#                     current_group.append(high_cost_points[i])
#                 else:
#                     issue_groups.append(current_group)
#                     current_group = [high_cost_points[i]]
#             issue_groups.append(current_group)
        
#         # 限制最多顯示 5 個建議，避免版面過長
#         for group in issue_groups:
#             start_k, end_k = group[0], group[-1]
            
#             # 對應到 Input 和 Reference 的幀索引
#             i_start, j_start = wp[start_k]
#             i_end, j_end = wp[end_k]
            
#             # 將 Input 幀索引轉換為時間 (秒)
#             time_start = i_start * TIME_PER_FRAME
#             time_end = i_end * TIME_PER_FRAME
            
#             # --- 局部音高 (Pitch) 分析 ---
#             # Chroma 特徵位於 features_input/ref 的前 12 行 (0-11)
#             chroma_input_seg = features_input[i_start:i_end+1, :12]
#             chroma_ref_seg = features_ref[j_start:j_end+1, :12]
            
#             pitch_suggestion = ""
#             if chroma_input_seg.size > 0 and chroma_ref_seg.size > 0:
#                 # 計算該區段內 Input 和 Ref 的主要音高 (Peak Chroma Bin)
#                 input_peak_bin = np.mean(np.argmax(chroma_input_seg, axis=1))
#                 ref_peak_bin = np.mean(np.argmax(chroma_ref_seg, axis=1))
#                 # pitch_diff 正值: 偏高 (Input Chroma Bin > Ref Chroma Bin)
#                 pitch_diff = input_peak_bin - ref_peak_bin 
                
#                 # 計算局部速度 (Local Speed) 差異
#                 frames_input_seg = i_end - i_start
#                 frames_ref_seg = j_end - j_start
#                 local_slope = frames_input_seg / (frames_ref_seg + 1e-6) # 避免除以零

#                 if abs(pitch_diff) > 0.8: # 音高差異超過約 0.8 個半音
#                     pitch_level = "偏高" if pitch_diff > 0 else "偏低"
#                     pitch_suggestion = f"音高明顯**{pitch_level}**，平均偏差約 {abs(pitch_diff):.1f} 個半音。"
#                 elif local_slope > 1.8:
#                     pitch_suggestion = "**時序嚴重拖沓**，您的演唱太慢了，需要更果斷地進入下一個樂句。"
#                 elif local_slope < 0.5:
#                     pitch_suggestion = "**時序嚴重超前**，您的演唱太快了，請仔細聆聽參考音訊的間隔。"
#                 else:
#                     pitch_suggestion = "音準或音色不匹配。請專注於該樂句的音準穩定性。"

#                 # TODO: why got 0 秒到 0 秒？
#                 feedback.append(f"• **時間 {time_start:.2f} 秒到 {time_end:.2f} 秒:** {pitch_suggestion}")
        
#         if len(issue_groups) > 5:
#             feedback.append("• ... (僅顯示前 5 個最顯著的問題點)")

#     return similarity_score, '\n'.join(feedback)

# # 核心邏輯 4: DTW 可視化 (DTW Visualization)
# def plot_dtw_path(D, wp, sr):
#     """繪製累積成本矩陣和最佳規整路徑。"""
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111)
    
#     # 顯示累積成本矩陣 (Accumulated Cost Matrix)
#     # D 的形狀為 (input_frames, ref_frames)。specshow 將第一維對應 y 軸，第二維對應 x 軸。
#     img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='time', ax=ax)
#     fig.colorbar(img, ax=ax, format='%+2.0f', label='累積成本 (Accumulated Cost)')
#     ax.set(title='DTW 累積成本矩陣與最佳路徑')
    
#     # 繪製最佳規整路徑 (Warping Path)
#     # wp[:, 0] 是 input frames, wp[:, 1] 是 reference frames.
#     # 我們繪製 (ref_time, input_time) 來對應 x, y 軸。
#     ax.plot(librosa.frames_to_time(wp[:, 1], sr=sr), librosa.frames_to_time(wp[:, 0], sr=sr), 
#             marker='o', color='red', linestyle='-', linewidth=2, alpha=0.5, 
#             label='最佳規整路徑 (Warping Path)')
    
#     ax.set_xlabel("參考音訊時間 (Reference Time)")
#     ax.set_ylabel("您的歌聲時間 (Input Time)")
#     ax.legend(loc='lower right')
    
#     # --- 修正 ---
#     # Gradio 的 gr.Plot 元件可以直接處理 matplotlib 的 Figure 物件，
#     # 這比手動轉換為 bytes 更簡潔。
#     # 我們不應在此處呼叫 plt.close(fig)，否則 Gradio 將無法渲染它。
#     return fig


# # Gradio 主要函數
# def singing_evaluator(input_audio_path, ref_audio_path):
#     """Gradio 接口的主要處理函數。"""
#     if not input_audio_path or not ref_audio_path:
#         raise gr.Error("請上傳或錄製您的歌聲和參考音訊檔案。")

#     try:
#         # 1. 特徵提取
#         # TODO: make sure all to the same sr like 16,000 and mono, and if input m4a, mp3, etc., convert to flac first
#         features_input, sr, y_input = extract_features(input_audio_path)
#         features_ref, sr, y_ref = extract_features(ref_audio_path)
        
#         # 2. DTW 對齊
#         # TODO: use DTW is not strict enough? I input 2 different songs, it still give me a score, so also output confidence of the song,
#         # confidence lower than xx% means different songs, output "different songs" message
#         D, wp = align_dtw(features_input, features_ref)
        
#         # 3. 分析並生成建議
#         # TODO: fix issue in analyze_results
#         similarity_score, feedback_text = analyze_results(wp, D, features_input, features_ref, sr)
        
#         # 4. 可視化 DTW 路徑
#         # dtw_plot = plot_dtw_path(D, wp, sr)
        
#         # 5. 返回結果
#         return similarity_score, feedback_text, input_audio_path, ref_audio_path
        
#     except gr.Error as e:
#         # 直接拋出 Gradio 的錯誤，使其能清晰地顯示在 UI 上
#         raise e
#     except Exception as e:
#         error_message = f"分析過程中發生未知錯誤: {e}"
#         print(error_message) # 在後端打印詳細錯誤以供調試
#         # 向用戶顯示一個更友好的錯誤訊息
#         raise gr.Error("分析失敗，請檢查您的音訊檔案是否有效，或稍後再試。")



# # --- Gradio 界面定義 ---

# # 描述和標題
# title = "🎙️ AI 歌聲相似性評估與輔助系統🎶"
# description = (
#     "上傳或**即時錄製**兩段音訊檔案（您的歌聲和參考音訊），系統將使用動態時間規整 (DTW) 技術對齊音軌，"
#     "分析節奏和音高差異，並提供具體的改進建議，例如幾秒到幾秒太快或音高偏低。"
# )

# with gr.Blocks(theme=gr.themes.Soft(), title=title) as demo:
#     gr.Markdown(f"# {title}")
#     gr.Markdown(description)

#     # 輸入區 (已加入 microphone 來源)
#     with gr.Row():
#         input_audio_upload = gr.Audio(
#             type="filepath", 
#             label="🎤 您的歌聲 (Input Audio)", 
#             sources=["upload", "microphone"] # 支援上傳和錄音
#         )
#         ref_audio_upload = gr.Audio(
#             type="filepath", 
#             label="🎧 參考音訊 (Reference Audio)", 
#             sources=["upload", "microphone"] # 支援上傳和錄音
#         )
    
#     analyze_btn = gr.Button("🚀 開始分析與評估", variant="primary")
    
#     # 輸出區
#     # 將 gr.Column 賦值給一個變數，以便在 .success() 中引用
#     result_outputs_group = gr.Column(visible=False) 
#     with result_outputs_group:
#         gr.Markdown("---")
#         gr.Markdown("## 📋 評估結果")
        
#         with gr.Row():
#             score_display = gr.Textbox(label="總體正規化相似度分數 (越低越相似)", show_label=True, scale=1)
            
#         # 將 Markdown 元件也賦值給一個變數
#         feedback_output = gr.Markdown("### 📜 具體改進建議")

#         dtw_plot_output = gr.Plot(label="DTW 規整路徑圖 (Warping Path Visualization)")
        
#         gr.Markdown("---")
#         gr.Markdown("## 🔊 同步播放 (請手動按下播放鍵)")
        

#         # TODO: 
#         # 1. 新增同時播放兩個原本音訊功能，也就是疊加
#         # 2. 新增播放修改後 input audio 功能
#         # 3. 儲存多個按鈕代表不同音訊，選擇要同時播放的幾個（按下去），後點選播放，則同時播放對應的幾個音訊
#         with gr.Row():
#             aligned_input_playback = gr.Audio(label="您的歌聲 (Input)", interactive=False, autoplay=False)
#             aligned_ref_playback = gr.Audio(label="參考音訊 (Reference)", interactive=False, autoplay=False)
#         gr.Markdown("*注意：Gradio 僅提供並排播放，您需要手動同時點擊播放以進行聽覺上的同步比對。DTW 結果是演算法上已對齊的。*")

#     # 綁定事件
#     # 點擊按鈕後，先將結果區域設為隱藏，以清除舊的結果
#     def hide_results():
#         return gr.Column(visible=False)

#     analyze_btn.click(
#         fn=hide_results,
#         inputs=None,
#         outputs=[result_outputs_group]
#     ).then(
#         fn=singing_evaluator,
#         inputs=[input_audio_upload, ref_audio_upload],
#         outputs=[score_display, feedback_output, aligned_input_playback, aligned_ref_playback]
#     ).success(
#         fn=lambda: gr.Column(visible=True), # 成功後再顯示結果
#         inputs=None,
#         outputs=result_outputs_group
#     )

# if __name__ == "__main__":
#     demo.launch(share=True)


import gradio as gr
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import tempfile
import os

# --- 全域設定 (Global Settings) ---
TARGET_SR = 16000 # 設定統一的取樣率以進行公平比較
HOP_LENGTH = 512 # Librosa 預設的 Frame Hop 長度

# 核心邏輯 1: 特徵提取 (Feature Extraction)
def extract_features(audio_path):
    """載入音訊、標準化處理，並提取 Chroma 和 RMS 特徵。"""
    if not audio_path or not os.path.exists(audio_path):
        raise gr.Error("請錄製或上傳有效的音訊檔案。")

    try:
        # 載入音訊，並重取樣至目標 SR，轉換為單聲道
        y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    except Exception as e:
        raise gr.Error(f"載入音訊檔案失敗: {e}")

    # 檢查音訊長度是否足夠進行分析
    MIN_SAMPLES = 2048 # FFT 運算需要一定的樣本數
    if len(y) < MIN_SAMPLES:
        raise gr.Error(f"音訊長度過短 ({len(y)/sr:.2f} 秒)，無法進行有效分析。")

    # 1. Chroma feature (音高/和聲內容)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)
    
    # 2. RMS (Root-Mean-Square Energy for volume)
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
    
    # 合併特徵並轉置 -> (N_frames, 13)
    features = np.vstack([chroma, rms])
    
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
    
    # 【已修改】將成本轉換為 0-100 的分數，越高越好
    # 使用指數衰減函數，k 值可調整轉換的靈敏度
    k = 2.0 
    similarity_score = 100 * np.exp(-k * normalized_cost)
    
    # 【新增】偵測是否為不同歌曲
    # 設定一個經驗閾值，若成本過高，可能代表是完全不同的歌曲
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
            
            # 【已修改】修復 "0 秒到 0 秒" 問題
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

# 【新增】核心邏輯 5: 混合音訊 (Mix Audio)
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


# Gradio 主要函數
def singing_evaluator(input_audio_path, ref_audio_path):
    """Gradio 介面的主要處理函數。"""
    if not input_audio_path or not ref_audio_path:
        raise gr.Error("請同時上傳或錄製您的歌聲和參考音訊。")

    try:
        # 1. 特徵提取
        features_input = extract_features(input_audio_path)
        features_ref = extract_features(ref_audio_path)
        
        # 2. DTW 對齊
        D, wp = align_dtw(features_input, features_ref)
        
        # 3. 分析並生成建議
        similarity_score, feedback_text = analyze_results(wp, D, features_input, features_ref)
        
        # 4. 可視化 DTW 路徑
        dtw_plot = plot_dtw_path(D, wp)

        # 5. 混合音訊
        mixed_audio_path = mix_audio(input_audio_path, ref_audio_path)
        
        # 6. 返回所有結果
        return similarity_score, feedback_text, dtw_plot, input_audio_path, ref_audio_path, mixed_audio_path
        
    except gr.Error as e:
        raise e # 直接拋出 Gradio 的錯誤
    except Exception as e:
        error_message = f"分析過程中發生未知錯誤: {e}"
        print(error_message) # 在後端日誌中打印詳細錯誤
        raise gr.Error("分析失敗，請檢查您的音訊檔案是否有效，或稍後再試。")

# --- Gradio 界面定義 ---
title = "🎙️ AI 歌聲相似性評估與輔助系統 🎶"
description = (
    "上傳或**即時錄製**您的歌聲和參考音訊，系統將使用動態時間規整 (DTW) 技術，"
    "從**音準、節奏**等多個維度進行分析，提供 **0-100 的相似度分數**和具體的改進建議。"
    "您還可以播放**疊加後**的音訊，直觀感受差異。"
)

with gr.Blocks(theme=gr.themes.Soft(), title=title) as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(description)

    with gr.Row():
        input_audio = gr.Audio(type="filepath", label="🎤 您的歌聲 (Input)", sources=["upload", "microphone"])
        ref_audio = gr.Audio(type="filepath", label="🎧 參考音訊 (Reference)", sources=["upload", "microphone"])
    
    analyze_btn = gr.Button("🚀 開始分析與評估", variant="primary")
    
    result_outputs_group = gr.Column(visible=False) 
    with result_outputs_group:
        gr.Markdown("---")
        gr.Markdown("## 📋 評估報告")
        
        with gr.Row():
            score_display = gr.Textbox(label="總體相似度分數 (0-100分，越高越好)", scale=1)
        
        feedback_output = gr.Markdown(label="### 📜 具體改進建議")

        dtw_plot_output = gr.Plot(label="DTW 規整路徑圖 (視覺化時間對齊)")
        
        gr.Markdown("---")
        gr.Markdown("## 🔊 音訊比對播放")
        
        with gr.Row():
            input_playback = gr.Audio(label="您的歌聲 (Input)")
            ref_playback = gr.Audio(label="參考音訊 (Reference)")
        
        mixed_playback = gr.Audio(label="🎛️ 疊加播放 (Mixed for Comparison)")
        gr.Markdown("*提示：播放「疊加播放」音軌，可以幫助您更清晰地聽出節奏和音準的差異。*")

    def run_analysis(input_audio_path, ref_audio_path):
        score, feedback, plot, _, _, mixed_path = singing_evaluator(input_audio_path, ref_audio_path)
        return {
            result_outputs_group: gr.Column(visible=True),
            score_display: score,
            feedback_output: feedback,
            dtw_plot_output: plot,
            input_playback: gr.Audio(value=input_audio_path, label="您的歌聲 (Input)"),
            ref_playback: gr.Audio(value=ref_audio_path, label="參考音訊 (Reference)"),
            mixed_playback: gr.Audio(value=mixed_path, label="🎛️ 疊加播放 (Mixed for Comparison)")
        }

    analyze_btn.click(
        fn=lambda: gr.Column(visible=False), # 點擊後先隱藏舊結果
        outputs=[result_outputs_group]
    ).then(
        fn=run_analysis,
        inputs=[input_audio, ref_audio],
        outputs=[result_outputs_group, score_display, feedback_output, dtw_plot_output, input_playback, ref_playback, mixed_playback]
    )

if __name__ == "__main__":
    demo.launch(share=True)

