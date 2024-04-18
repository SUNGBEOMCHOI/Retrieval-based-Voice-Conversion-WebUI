import os
import sys
import argparse
import logging
from time import sleep
from subprocess import Popen
import threading

# Directory setup
# now_dir = os.getcwd()
# sys.path.append(now_dir)

# Retrieval-based-Voice-Conversion-WebUI 경로를 sys.path에 추가
project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_path)

from configs.config import Config

# Set logging levels
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Sampling rate dictionary
sr_dict = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}

# Configuration from external file
config = Config()

def if_done(done, p):
    p.wait()
    done[0] = True

def preprocess_dataset(trainset_dir, exp_dir, sr, n_p):
    os.makedirs(trainset_dir, exist_ok=True)
    
    sr = sr_dict[sr]
    os.makedirs(exp_dir, exist_ok=True)
    log_file = f"{exp_dir}/preprocess.log"
    f = open(log_file, "a+")
    f.close()


    
    cmd = f'"{config.python_cmd}" {project_path}/infer/modules/train/preprocess.py "{trainset_dir}" {sr} {n_p} "{exp_dir}" {config.noparallel} {config.preprocess_per}'
    logger.info("Execute: " + cmd)
    p = Popen(cmd, shell=True)
    done = [False]
    threading.Thread(target=if_done, args=(done, p)).start()

def arg_parse():
    parser = argparse.ArgumentParser(description="Preprocess audio data for training.")
    parser.add_argument('--trainset_dir', type=str, default="/home/choi/desktop/rvc/ai/data/user2/input/speaker",
                        help="Directory where the training dataset is stored")
    parser.add_argument('--exp_dir', type=str, default="/home/choi/desktop/rvc/ai/data/user2/output/trained_model",
                        help="Directory where the experiment's outputs are saved")
    parser.add_argument('--sampling_rate', type=str, choices=['32k', '40k', '48k'], default="40k",
                        help="Sampling rate for the audio processing")
    parser.add_argument('--n_p', type=int, default=8,
                        help="Number of processes to use")
    return parser.parse_args()

def main():
    args = arg_parse()
    preprocess_dataset(args.trainset_dir, args.exp_dir, args.sampling_rate, args.n_p)

if __name__ == "__main__":
    main()
