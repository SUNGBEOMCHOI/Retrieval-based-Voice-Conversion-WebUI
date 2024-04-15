import os
import sys
import logging
import traceback
import argparse
import torch
import ffmpeg

# Adding current working directory to path
now_dir = os.getcwd()
sys.path.append(now_dir)

from configs.config import Config
from infer.modules.uvr5.mdxnet import MDXNetDereverb
from infer.modules.uvr5.vr import AudioPre, AudioPreDeEcho

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Adjust as necessary

config = Config()

def initialize_pre_fun(model_name, config, agg):
    weight_root = os.getenv("weight_uvr5_root", "/home/choi/desktop/rvc/ai/Retrieval-based-Voice-Conversion-WebUI/assets/uvr5_weights")
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

def uvr(model_name, inp_path, save_root_vocal, paths, save_root_ins, agg, format0):
    os.makedirs(save_root_vocal, exist_ok=True)
    os.makedirs(save_root_ins, exist_ok=True)

    infos = []
    pre_fun = None
    try:
        pre_fun = initialize_pre_fun(model_name, config, agg)
        need_reformat = 1
        done = 0
        try:
            info = ffmpeg.probe(inp_path, cmd="ffprobe")
            if info["streams"][0]["channels"] == 2 and info["streams"][0]["sample_rate"] == "44100":
                need_reformat = 0
                pre_fun._path_audio_(inp_path, save_root_ins, save_root_vocal, format0)
                done = 1
        except Exception as e:
            infos.append(f"Error in probing the file: {str(e)}")
            traceback.print_exc()

        if need_reformat == 1:
            tmp_path = f"{os.environ.get('TEMP', '/tmp')}/{os.path.basename(inp_path)}.reformatted.wav"
            os.system(f"ffmpeg -i {inp_path} -vn -acodec pcm_s16le -ac 2 -ar 44100 {tmp_path} -y")
            inp_path = tmp_path

        if done == 0:
            pre_fun._path_audio_(inp_path, save_root_ins, save_root_vocal, format0)
            infos.append(f"{os.path.basename(inp_path)}->Success")
        else:
            infos.append("No reformat needed and processing completed.")

    except Exception as e:
        infos.append(f"Exception during audio processing: {str(e)}")
        traceback.print_exc()
    finally:
        clean_up(pre_fun)
    return "\n".join(infos)

def clean_up(pre_fun):
    try:
        if pre_fun:
            if hasattr(pre_fun, 'model'):
                del pre_fun.model
            if hasattr(pre_fun, 'pred') and hasattr(pre_fun.pred, 'model'):
                del pre_fun.pred.model
    except Exception as e:
        logger.error(f"Failed to clean up resources: {str(e)}")
        traceback.print_exc()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("Executed torch.cuda.empty_cache()")

def arg_parse():
    parser = argparse.ArgumentParser(description="Audio processing with UVR model")
    parser.add_argument("--model_name", type=str, default="HP5_only_main_vocal", help="Model name for processing")
    parser.add_argument("--inp_path", type=str, default="/home/choi/desktop/rvc/ai/data/user1/input/music/origin_music.mp3", help="Input path of the audio file")
    parser.add_argument("--save_root_vocal", type=str, default="/home/choi/desktop/rvc/ai/data/user1/output/music", help="Output path for vocal")
    parser.add_argument("--save_root_ins", type=str, default="/home/choi/desktop/rvc/ai/data/user1/output/music", help="Output path for instruments")
    parser.add_argument("--agg", type=int, default=0, help="Aggregation parameter")
    parser.add_argument("--format0", type=str, default="wav", help="Output audio format")
    return parser.parse_args()

def main():
    args = arg_parse()
    result = uvr(args.model_name, args.inp_path, args.save_root_vocal, [], args.save_root_ins, args.agg, args.format0)
    print(result)

if __name__ == "__main__":
    main()
