#!/bin/bash

printf "working dir is %s\n" "$PWD"
echo "downloading requirement aria2 check."

if ! command -v aria2c > /dev/null 2>&1; then
    echo "failed. please install aria2"
    exit 1
fi

echo "dir check start."

check_dir() {
    [ -d "$1" ] && printf "dir %s checked\n" "$1" || \
    (printf "failed. generating dir %s\n" "$1" && mkdir -p "$1")
}

check_dir "./assets/pretrained"
check_dir "./assets/pretrained_v2"
check_dir "./assets/uvr5_weights"
check_dir "./assets/uvr5_weights/onnx_dereverb_By_FoxJoy"

echo "dir check finished."

# List of files to download
files=(
  "pretrained/D32k.pth"
  "pretrained/D40k.pth"
  "pretrained/D48k.pth"
  "pretrained/G32k.pth"
  "pretrained/G40k.pth"
  "pretrained/G48k.pth"
  "pretrained_v2/f0D40k.pth"
  "pretrained_v2/f0G40k.pth"
  "pretrained_v2/D40k.pth"
  "pretrained_v2/G40k.pth"
  "uvr5_weights/HP2_all_vocals.pth"
  "uvr5_weights/HP3_all_vocals.pth"
  "uvr5_weights/HP5_only_main_vocal.pth"
  "uvr5_weights/VR-DeEchoAggressive.pth"
  "uvr5_weights/VR-DeEchoDeReverb.pth"
  "uvr5_weights/VR-DeEchoNormal.pth"
  "uvr5_weights/onnx_dereverb_By_FoxJoy/vocals.onnx"
)

for file in "${files[@]}"; do
    dir=$(dirname "$file")
    basename=$(basename "$file")
    url="https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/$file"
    download_path="./assets/$dir"
    mkdir -p "$download_path" # Ensure directory exists
    if [ ! -f "$download_path/$basename" ]; then
        echo "Downloading $basename to $download_path"
        aria2c --console-log-level=error -c -x 16 -s 16 -k 1M "$url" -d "$download_path" -o "$basename"
    else
        echo "$basename already exists in $download_path"
    fi
done

special_files=(
    "rmvpe/rmvpe.pt"
    "hubert/hubert_base.pt"
)

for file in "${special_files[@]}"; do
    dir=$(dirname "$file")
    basename=$(basename "$file")
    printf "checking %s\n" "$basename"
    if [ -f "./assets/$dir/$basename" ]; then
        printf "%s in ./assets/%s checked.\n" "$basename" "$dir"
    else
        echo "failed. starting download from huggingface."
        if command -v aria2c > /dev/null 2>&1; then
            aria2c --console-log-level=error -c -x 16 -s 16 -k 1M "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/$basename" -d "./assets/$dir" -o "$basename"
            if [ -f "./assets/$dir/$basename" ]; then
                echo "download successful."
            else
                echo "please try again!"
                exit 1
            fi
        else
            echo "aria2c command not found. Please install aria2 and try again."
            exit 1
        fi
    fi
done

echo "All required files have been checked or downloaded."
