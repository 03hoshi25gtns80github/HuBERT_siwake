"""
    ../testフォルダ内の動画を.wavファイルに変換
"""

import os
from moviepy.editor import VideoFileClip

def convert_videos_to_wav(input_folder, output_folder):
    # 入力フォルダ内のすべてのファイルを取得
    for filename in os.listdir(input_folder):
        if filename.endswith((".MTS", ".mp4")):
            input_path = os.path.join(input_folder, filename)
            output_filename = os.path.splitext(filename)[0] + ".wav"
            output_path = os.path.join(output_folder, output_filename)
            
            # 動画ファイルを読み込み
            video = VideoFileClip(input_path)
            
            # 動画の長さを確認
            if video.duration < 7:
                print(f"Skipped {input_path} because it is shorter than 7 seconds")
                continue
            
            # 最初の7秒の音声を抽出
            audio = video.audio.subclip(0, 7)
            
            # 音声を.wavファイルとして保存
            audio.write_audiofile(output_path, codec='pcm_s16le')
            print(f"Converted {input_path} to {output_path}")

def main():
    test_folder = "../test"
    wav_folder = "../wav"

    # フォルダが存在しない場合は作成
    if not os.path.exists(wav_folder):
        os.makedirs(wav_folder)

    # 動画ファイルから音声ファイルを作成
    convert_videos_to_wav(test_folder, wav_folder)

if __name__ == "__main__":
    main()
