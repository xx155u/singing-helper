# usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trim_audio.py  ‒ 依指定起訖秒數裁切音訊
usage 示例：
  # 取 5.0 s ~ 12.3 s 這段
  python3 trim_audio.py --input_audio_path in.wav --output_audio_path out.wav \
                        --trim_from_sec 5.0 --drop_from_sec 12.3
"""
import argparse
import os
import sys
from datetime import datetime

import numpy as np
import soundfile as sf


def trim_audio(input_path: str,
               output_path: str,
               trim_from_sec: float | None = None,
               drop_from_sec: float | None = None) -> bool:
    """
    裁切音訊，可同時指定起點與終點。
    - 兩者皆給：保留 [trim_from_sec, drop_from_sec) 區間
    - 只給 trim_from_sec：保留起點之後全部
    - 只給 drop_from_sec：保留開頭到終點
    """
    try:
        if not os.path.exists(input_path):
            print(f"錯誤: 找不到輸入檔案 '{input_path}'")
            return False

        audio, sample_rate = sf.read(input_path)
        original_duration = len(audio) / sample_rate
        print(f"原始音訊長度: {original_duration:.2f} 秒")

        # 預設值
        start_point = 0
        end_point = len(audio)

        # 起點處理
        if trim_from_sec is not None:
            if trim_from_sec < 0:
                print("錯誤: trim_from_sec 不能為負值")
                return False
            start_point = int(trim_from_sec * sample_rate)

        # 終點處理
        if drop_from_sec is not None:
            if drop_from_sec <= 0:
                print("錯誤: drop_from_sec 必須大於 0")
                return False
            end_point = int(drop_from_sec * sample_rate)

        # 邊界與順序檢查
        if start_point >= end_point:
            print("錯誤: trim_from_sec 必須小於 drop_from_sec")
            return False
        if end_point > len(audio):
            print("警告: drop_from_sec 超出音訊長度，將取到檔尾")
            end_point = len(audio)

        trimmed_audio = audio[start_point:end_point]
        trimmed_duration = len(trimmed_audio) / sample_rate

        # 輸出
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, trimmed_audio, sample_rate)

        print("成功裁切音訊：")
        print(f"  - 保留範圍: {start_point/sample_rate:.2f}s ~ {end_point/sample_rate:.2f}s")
        print(f"  - 裁切後長度: {trimmed_duration:.2f} 秒")
        print(f"  - 輸出檔案: {output_path}")
        return True

    except Exception as e:
        print(f"處理音訊時發生錯誤: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="裁切音訊，可同時指定起點 (--trim_from_sec) 與終點 (--drop_from_sec)")
    parser.add_argument("--input_audio_path", required=True, help="輸入音訊路徑")
    parser.add_argument("--output_audio_path", required=True, help="輸出音訊路徑")
    parser.add_argument("--trim_from_sec", type=float, help="保留區段起點 (秒)")
    parser.add_argument("--drop_from_sec", type=float, help="保留區段終點 (秒)")

    args = parser.parse_args()

    # 至少要指定一個
    if args.trim_from_sec is None and args.drop_from_sec is None:
        parser.error("必須指定 --trim_from_sec、--drop_from_sec 其中一個")

    start_time = datetime.now()
    print(f"開始處理: {start_time:%Y-%m-%d %H:%M:%S}")

    success = trim_audio(args.input_audio_path,
                         args.output_audio_path,
                         args.trim_from_sec,
                         args.drop_from_sec)

    end_time = datetime.now()
    print(f"結束處理: {end_time:%Y-%m-%d %H:%M:%S}")
    print(f"總花費時間: {(end_time - start_time).total_seconds():.2f} 秒")
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()


"""
python /Users/wtforbes/Library/CloudStorage/Dropbox/singing-helper/trim_audio.py \
    --input_audio_path /Users/wtforbes/Library/CloudStorage/Dropbox/singing-helper/關於小熊.flac \
    --output_audio_path /Users/wtforbes/Library/CloudStorage/Dropbox/singing-helper/關於小熊_10.flac \
    --trim_from_sec 17 --drop_from_sec 27

python /Users/wtforbes/Library/CloudStorage/Dropbox/singing-helper/trim_audio.py \
    --input_audio_path /Users/wtforbes/Library/CloudStorage/Dropbox/singing-helper/關於小熊_mod.flac \
    --output_audio_path /Users/wtforbes/Library/CloudStorage/Dropbox/singing-helper/關於小熊_10_mod.flac \
    --trim_from_sec 17 --drop_from_sec 27


python3 /home/cky/Taiwan-Whisper/tts_process/trim_audio.py \
    --input_audio_path /home/cky/ytb_list/youtube_downloads_speaker/ruanjiao.wav \
    --output_audio_path /home/cky/forbes/debug_area/ruanjiao.wav \
    --trim_from_sec 36.7 --drop_from_sec 46.5


# 從開頭修剪 (保留 0.3 秒後的內容)
python3 /home/cky/Taiwan-Whisper/tts_process/trim_audio.py \
    --input_audio_path /home/cky/BreezyVoice/results/out_forbes.wav \
    --output_audio_path /home/cky/forbes/debug_area/blur_0.wav \
    --trim_from_sec 0.4

# 從結尾修剪 (丟棄 5.7 秒後的內容)
python3 /home/cky/Taiwan-Whisper/tts_process/trim_audio.py \
    --input_audio_path /home/cky/selected_cs/Psyman/rGwSRdYhbJw/Psyman_rGwSRdYhbJw_0_84.wav \
    --output_audio_path /home/cky/selected_cs/Psyman/rGwSRdYhbJw/Psyman_rGwSRdYhbJw_0_84_test.wav \
    --drop_from_sec 16.45
"""

"""
id      text    audio
04a92635-8258-49b3-960a-5f806468aee2    <|0.00|>新加入可換線設計 原裝配上6<|2.34|><|2.34|>藉由實踐公司使命<|4.26|><|4.26|>我們幫助他人實現他們的目標<|6.42|><|6.42|>當然也可能出現類似八點檔那般為了分遺產故意拖著不離婚的狗血情況<|13.90|><|13.90|>雖然機率很低但並非為零<|16.62|><|16.62|>抄去不是等於惡搞自己嗎<|19.82|><|19.82|>台中和台北等地的應用和影響<|22.52|><|22.52|>白巫師<|24.30|><|24.30|>能使我忠於我的信念生活<|27.32|><|endoftext|>    /work/forbes110/TTS_pack/final_audio/fine_grained_fineweb_1_long_fm/04a92635-8258-49b3-960a-5f806468aee2.wav
common_voice_en_18963008        <|0.00|>This condition is sufficient but not necessary.<|4.12|><|endoftext|>    /work/forbes110/TTS_pack/local_train_wav/CV17EN_v2/common_voice_en_18963008.wav
common_voice_en_25922257        <|0.00|>The modern cultural space Palmer’s Rock was set up there.<|4.78|><|endoftext|>  /work/forbes110/TTS_pack/local_train_wav/CV17EN_v2/common_voice_en_25922257.wav
common_voice_en_22777664        <|0.00|>The also supports coastal dune systems, lagoons and coastal vegetation.<|7.54|><|endoftext|>    /work/forbes110/TTS_pack/local_train_wav/CV17EN_v2/common_voice_en_22777664.wav
994dca4b-1843-4855-ada0-d7ef305ba0b7    <|0.00|>不只讓家長可以放心上班<|3.72|><|3.72|>更是讓孩子們可以自主學習<|5.56|><|5.56|>也到戶外玩中學學中玩<|7.16|><|7.16|>香港仔運動場換好衫<|9.22|><|9.22|>做好熱身出發<|10.26|><|10.26|>是另一次資訊工業革命<|13.02|><|13.02|>能幫助人類產業進化<|15.14|><|15.14|>這些限制反映了塔位使用權的債權性質<|20.50|><|20.50|>即使用人並未擁有塔位本身的所有權<|23.54|><|23.54|>而是依據契約享有特定的使用權益<|28.64|><|28.64|><|endoftext|> /work/forbes110/TTS_pack/final_audio/fine_grained_fineweb_1_long_fm/994dca4b-1843-4855-ada0-d7ef305ba0b7.wav
common_voice_en_31923211        <|0.00|>He ultimately rated the album two out of five stars.<|5.22|><|endoftext|>       /work/forbes110/TTS_pack/local_train_wav/CV17EN_v2/common_voice_en_31923211.wav
common_voice_en_25216810        <|0.00|>He was awarded two papal knighthoods and was a Knight of Malta.<|6.94|><|endoftext|>    /work/forbes110/TTS_pack/local_train_wav/CV17EN_v2/common_voice_en_25216810.wav
d3edee2b-6272-4cd4-a5b0-cef887724fd6    <|0.00|>水上市場離曼谷的車程約需2小時<|4.60|><|4.60|>交通不怎麼方便自由行的人應該很少來這裡吧<|9.04|><|9.04|>未使用前髮絲很細<|11.84|><|11.84|>中間幾乎沒有頭髮<|13.48|><|13.48|>而且頭皮上有毛囊炎的傷口<|16.18|><|16.18|>頭皮上油脂分泌明顯旺盛<|18.96|><|18.96|>過膝裙中尤其受歡迎的百褶裙款式<|23.34|><|23.34|>近兩季最流行的金屬材質百褶裙與秋季必備針織衫也是絕配<|29.54|><|29.54|><|endoftext|> /work/forbes110/TTS_pack/final_audio/fine_grained_fineweb_1_long_fm/d3edee2b-6272-4cd4-a5b0-cef887724fd6.wav
common_voice_en_19037364        <|0.00|>Shortly afterward, Ramirez stated that a physician had unknowingly prescribed a banned medication.<|8.26|><|endoftext|> /work/forbes110/TTS_pack/local_train_wav/CV17EN_v2/common_voice_en_19037364.wav
"""