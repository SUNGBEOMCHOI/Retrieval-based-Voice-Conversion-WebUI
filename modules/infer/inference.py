import argparse
import os
import sys

# now_dir = os.getcwd()
# sys.path.append(now_dir)

# Retrieval-based-Voice-Conversion-WebUI 경로를 sys.path에 추가
project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_path)

from dotenv import load_dotenv
from scipy.io import wavfile
import ffmpeg

from configs.config import Config
from infer.modules.vc.modules import VC


def arg_parse() -> tuple:
    parser = argparse.ArgumentParser()
    parser.add_argument("--f0up_key", type=int, default=0)
    parser.add_argument("--input_path", type=str, help="input path", default="/home/choi/desktop/rvc/ai/data/user2/output/music/vocal_origin_music.mp3.wav")
    parser.add_argument("--index_path", type=str, help="index path", default="/home/choi/desktop/rvc/ai/data/user2/output/trained_model/trained_index.index")
    parser.add_argument("--f0method", type=str, default="rmvpe", help="harvest or pm")
    parser.add_argument("--opt_path", type=str, help="opt path", default="/home/choi/desktop/rvc/ai/data/user2/output/cover/output.wav")
    parser.add_argument("--model_name", type=str, help="store in assets/weight_root", default="/home/choi/desktop/rvc/ai/data/user2/output/trained_model/trained_voice.pth")
    parser.add_argument("--index_rate", type=float, default=0.66, help="index rate")
    parser.add_argument("--device", type=str, help="device")
    parser.add_argument("--is_half", type=bool, help="use half -> True")
    parser.add_argument("--filter_radius", type=int, default=3, help="filter radius")
    parser.add_argument("--resample_sr", type=int, default=0, help="resample sr")
    parser.add_argument("--rms_mix_rate", type=float, default=1, help="rms mix rate")
    parser.add_argument("--protect", type=float, default=0.33, help="protect")

    args = parser.parse_args()
    sys.argv = sys.argv[:1]

    return args

def rvc_inference(voice_model, voice_model_path, input_path, output_path, index_path, inference_args):
    load_dotenv()
    f0up_key = inference_args.get('f0up_key', 0)
    f0method = inference_args.get('f0method', "rmvpe")
    index_rate = inference_args.get('index_rate', 0.66)
    filter_radius = inference_args.get('filter_radius', 3)
    resample_sr = inference_args.get('resample_sr', 0)
    rms_mix_rate = inference_args.get('rms_mix_rate', 1)
    protect = inference_args.get('protect', 0.33)

    voice_model.get_vc(voice_model_path)
    _, wav_opt = voice_model.vc_single(
        0,
        input_path,
        f0up_key,
        None,
        f0method,
        index_path,
        None,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    wavfile.write(output_path, wav_opt[0], wav_opt[1])
    return output_path

if __name__ == "__main__":
    args = arg_parse()
    # main(args)
    vc = VC(Config())
    rvc_inference(voice_model=vc, voice_model_path=args.model_name, input_path=args.input_path, output_path=args.opt_path, index_path=args.index_path, inference_args=vars(args))
