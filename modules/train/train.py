import argparse
import os
import sys
import logging
from subprocess import Popen
import pathlib
import json
from random import shuffle
from time import sleep

# Add the current directory to the system path
# now_dir = os.getcwd()
# sys.path.append(now_dir)

# Retrieval-based-Voice-Conversion-WebUI 경로를 sys.path에 추가
project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_path)

# Import custom modules and configurations
from i18n.i18n import I18nAuto
from configs.config import Config

# Set up logging levels for external libraries
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Set up a logger for this module
logger = logging.getLogger(__name__)

# Initialize configuration and internationalization
config = Config()
i18n = I18nAuto()

def click_train(exp_dir1, args):    
    sr2=args.get("sampling_rate", "40k")
    if_f0_3=args.get("if_f0_3", True)
    spk_id5=args.get("spk_id", "0")
    save_epoch10=args.get("save_epoch", 5)
    total_epoch11=args.get("total_epoch", 10)
    batch_size12=args.get("batch_size", 4)
    if_save_latest13=args.get("if_save_latest", "아니오")
    pretrained_G14=args.get("pretrained_g", f"{project_path}/assets/pretrained_v2/f0G40k.pth")
    pretrained_D15=args.get("pretrained_d", f"{project_path}/assets/pretrained_v2/f0D40k.pth")
    gpus16=args.get("gpus", "0")
    if_cache_gpu17=args.get("if_cache_gpu", "아니오")
    if_save_every_weights18=args.get("if_save_every_weights", "아니오")
    version19=args.get("version", "v2")

    exp_dir = "%s" % (exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    gt_wavs_dir = "%s/0_gt_wavs" % (exp_dir)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    if if_f0_3:
        f0_dir = "%s/2a_f0" % (exp_dir)
        f0nsf_dir = "%s/2b-f0nsf" % (exp_dir)
        names = (
            set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
            & set([name.split(".")[0] for name in os.listdir(feature_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
        )
    else:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(feature_dir)]
        )
    opt = []
    for name in names:
        if if_f0_3:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    f0_dir.replace("\\", "\\\\"),
                    name,
                    f0nsf_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
        else:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
    fea_dim = 256 if version19 == "v1" else 768
    if if_f0_3:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
                % (project_path, sr2, project_path, fea_dim, project_path, project_path, spk_id5)
            )
    else:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s"
                % (project_path, sr2, project_path, fea_dim, spk_id5)
            )
    shuffle(opt)
    with open("%s/filelist.txt" % exp_dir, "w") as f:
        f.write("\n".join(opt))
    logger.debug("Write filelist done")
    # 生成config#无需生成config
    # cmd = python_cmd + " train_nsf_sim_cache_sid_load_pretrain.py -e mi-test -sr 40k -f0 1 -bs 4 -g 0 -te 10 -se 5 -pg pretrained/f0G40k.pth -pd pretrained/f0D40k.pth -l 1 -c 0"
    logger.info("Use gpus: %s", str(gpus16))
    if pretrained_G14 == "":
        logger.info("No pretrained Generator")
    if pretrained_D15 == "":
        logger.info("No pretrained Discriminator")
    if version19 == "v1" or sr2 == "40k":
        config_path = "v1/%s.json" % sr2
    else:
        config_path = "v2/%s.json" % sr2
    config_save_path = os.path.join(exp_dir, "config.json")
    if not pathlib.Path(config_save_path).exists():
        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(
                config.json_config[config_path],
                f,
                ensure_ascii=False,
                indent=4,
                sort_keys=True,
            )
            f.write("\n")
    if gpus16:
        cmd = (
            '"%s" infer/modules/train/train.py -e "%s" -sr %s -f0 %s -bs %s -g %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s'
            % (
                config.python_cmd,
                exp_dir1,
                sr2,
                1 if if_f0_3 else 0,
                batch_size12,
                gpus16,
                total_epoch11,
                save_epoch10,
                "-pg %s" % pretrained_G14 if pretrained_G14 != "" else "",
                "-pd %s" % pretrained_D15 if pretrained_D15 != "" else "",
                1 if if_save_latest13 == i18n("是") else 0,
                1 if if_cache_gpu17 == i18n("是") else 0,
                1 if if_save_every_weights18 == i18n("是") else 0,
                version19,
            )
        )
    else:
        cmd = (
            '"%s" infer/modules/train/train.py -e "%s" -sr %s -f0 %s -bs %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s'
            % (
                config.python_cmd,
                exp_dir1,
                sr2,
                1 if if_f0_3 else 0,
                batch_size12,
                total_epoch11,
                save_epoch10,
                "-pg %s" % pretrained_G14 if pretrained_G14 != "" else "",
                "-pd %s" % pretrained_D15 if pretrained_D15 != "" else "",
                1 if if_save_latest13 == i18n("是") else 0,
                1 if if_cache_gpu17 == i18n("是") else 0,
                1 if if_save_every_weights18 == i18n("是") else 0,
                version19,
            )
        )
    logger.info("Execute: " + cmd)
    p = Popen(cmd, shell=True, cwd=project_path)
    p.wait()
    trained_voice_path = os.path.join(exp_dir, "trained_voice.pth")
    return trained_voice_path

def arg_parse():
    parser = argparse.ArgumentParser(description="Run the training process with specified parameters.")
    parser.add_argument('--exp_dir', type=str, default="/home/choi/desktop/rvc/ai/data/user2/output/trained_model")
    parser.add_argument('--sampling_rate', type=str, default="40k")
    parser.add_argument('--if_f0_3', type=bool, default=True)
    parser.add_argument('--spk_id', type=str, default="0")
    parser.add_argument('--save_epoch', type=int, default=5)
    parser.add_argument('--total_epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--if_save_latest', type=str, default="아니오")
    parser.add_argument('--pretrained_G', type=str, default=f"{project_path}/assets/pretrained_v2/f0G40k.pth")
    parser.add_argument('--pretrained_D', type=str, default=f"{project_path}/assets/pretrained_v2/f0D40k.pth")
    parser.add_argument('--gpus', type=str, default="0")
    parser.add_argument('--if_cache_gpu', type=str, default="아니오")
    parser.add_argument('--if_save_every_weights', type=str, default="아니오")
    parser.add_argument('--version', type=str, default="v2")
    return parser.parse_args()

def main():
    args = arg_parse()
    trained_voice_path = click_train(exp_dir1=args.exp_dir, args=vars(args))
    print(trained_voice_path)

if __name__ == "__main__":
    main()