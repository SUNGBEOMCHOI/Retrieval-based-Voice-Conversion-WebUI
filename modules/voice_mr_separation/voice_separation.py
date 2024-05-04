import os
import sys
import logging
import traceback
import argparse
import torch
import ffmpeg

# Adding current working directory to path
# now_dir = os.getcwd()
# sys.path.append(now_dir)

# Retrieval-based-Voice-Conversion-WebUI 경로를 sys.path에 추가
project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_path)

from configs.config import Config
from infer.modules.uvr5.mdxnet import MDXNetDereverb
from infer.modules.uvr5.vr import AudioPre, AudioPreDeEcho

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Adjust as necessary

def initialize_voice_separation_model(model_name, agg):
    config = Config()
    weight_root = os.getenv("weight_uvr5_root", f"{project_path}/assets/uvr5_weights")
    if model_name == "onnx_dereverb_By_FoxJoy":
        return MDXNetDereverb(15, config.device)
    else:
        Cls = AudioPre if "DeEcho" not in model_name else AudioPreDeEcho
        return Cls(
            agg=int(agg),
            model_path=os.path.join(weight_root, model_name + ".pth"),
            device=config.device,
            is_half=config.is_half,
        )

def uvr(model, inp_path, save_root_vocal, save_root_ins, agg, format0):
    os.makedirs(save_root_vocal, exist_ok=True)
    os.makedirs(save_root_ins, exist_ok=True)

    infos = []
    need_reformat = 1
    done = 0
    try:
        info = ffmpeg.probe(inp_path, cmd="ffprobe")
        if info["streams"][0]["channels"] == 2 and info["streams"][0]["sample_rate"] == "44100":
            need_reformat = 0
            model._path_audio_(inp_path, save_root_ins, save_root_vocal, format0)
            done = 1
    except Exception as e:
        infos.append(f"Error in probing the file: {str(e)}")
        traceback.print_exc()

    if need_reformat == 1:
        tmp_path = f"{os.environ.get('TEMP', '/tmp')}/{os.path.basename(inp_path)}.reformatted.wav"
        os.system(f"ffmpeg -i {inp_path} -vn -acodec pcm_s16le -ac 2 -ar 44100 {tmp_path} -y")
        inp_path = tmp_path

    if done == 0:
        model._path_audio_(inp_path, save_root_ins, save_root_vocal, format0)
        infos.append(f"{os.path.basename(inp_path)}->Success")
    else:
        infos.append("No reformat needed and processing completed.")
        
    return "\n".join(infos)

def clean_up(model):
    try:
        if model:
            if hasattr(model, 'model'):
                del model.model
            if hasattr(model, 'pred') and hasattr(model.pred, 'model'):
                del model.pred.model
    except Exception as e:
        logger.error(f"Failed to clean up resources: {str(e)}")
        traceback.print_exc()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("Executed torch.cuda.empty_cache()")

def arg_parse():
    parser = argparse.ArgumentParser(description="Audio processing with UVR model")
    parser.add_argument("--model_name", type=str, default="HP5_only_main_vocal", help="Model name for processing")
    parser.add_argument("--inp_path", type=str, default="/home/choi/desktop/rvc/ai/data/user2/input/music/origin_music.mp3", help="Input path of the audio file")
    parser.add_argument("--save_root_vocal", type=str, default="/home/choi/desktop/rvc/ai/data/user2/output/music", help="Output path for vocal")
    parser.add_argument("--save_root_ins", type=str, default="/home/choi/desktop/rvc/ai/data/user2/output/music", help="Output path for instruments")
    parser.add_argument("--agg", type=int, default=10, help="Aggregation parameter")
    parser.add_argument("--format0", type=str, default="wav", help="Output audio format")
    return parser.parse_args()

def main():
    args = arg_parse()
    model = initialize_voice_separation_model(args)
    try:
        uvr(model, args.inp_path, args.save_root_vocal, args.save_root_ins, args.agg, args.format0)
    except Exception as e:
        logger.error(f"Failed to process audio: {str(e)}")
        traceback.print_exc()
    finally:
        clean_up(model)

if __name__ == "__main__":
    main()
