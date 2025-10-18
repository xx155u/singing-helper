# 範例連結: https://youtu.be/t47UAKBHHsI?si=JsKt8nCRTG7R13Fy
VIDEO_URL="https://youtu.be/t47UAKBHHsI?si=JsKt8nCRTG7R13Fy"
OUTPUT_FILENAME="%(title)s.flac"

yt-dlp \
    -x \
    --audio-format flac \
    -o "$OUTPUT_FILENAME" \
    --postprocessor-args "-ac 1 -ar 16000" \
    "$VIDEO_URL"


# bash file
"""
python file_name.py --pitch 5 --speed 1.2 --input input.wav --output output.wav

wav, flac, m4a, mp3
"""