import os
import sys
# now_dir = os.getcwd()
# sys.path.append(now_dir)

# Retrieval-based-Voice-Conversion-WebUI 경로를 sys.path에 추가
project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_path)

import argparse
import logging
from time import sleep
from subprocess import Popen
import threading

from configs.config import Config

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

config = Config()

def if_done(done, p):
    while 1:
        if p.poll() is None:
            sleep(0.5)
        else:
            break
    done[0] = True

def if_done_multi(done, ps):
    while 1:
        # poll==None代表进程未结束
        # 只要有一个进程未结束都不停
        flag = 1
        for p in ps:
            if p.poll() is None:
                flag = 0
                sleep(0.5)
                break
        if flag == 1:
            break
    done[0] = True

def extract_f0_feature(exp_dir, args):
    gpus = str(args.get("gpus", "0"))
    n_p = args.get("n_p", 8)
    f0method = args.get("f0method", "rmvpe_gpu")
    if_f0 = args.get("if_f0_3", True)
    version19 = args.get("version19", "v2")
    gpus_rmvpe = args.get("gpus_rmvpe", "0-0")

    gpus = gpus.split("-")
    os.makedirs(exp_dir, exist_ok=True)
    f = open("%s/extract_f0_feature.log" % exp_dir, "w")
    f.close()
    if if_f0:
        if f0method != "rmvpe_gpu":
            cmd = (
                '"%s" infer/modules/train/extract/extract_f0_print.py "%s" %s %s'
                % (
                    config.python_cmd,
                    project_path,
                    exp_dir,
                    n_p,
                    f0method,
                )
            )
            logger.info("Execute: " + cmd)
            p = Popen(
                cmd, shell=True, cwd=project_path
            )  # , stdin=PIPE, stdout=PIPE,stderr=PIPE
            # 煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
            done = [False]
            threading.Thread(
                target=if_done,
                args=(
                    done,
                    p,
                ),
            ).start()
        else:
            if gpus_rmvpe != "-":
                gpus_rmvpe = gpus_rmvpe.split("-")
                leng = len(gpus_rmvpe)
                ps = []
                for idx, n_g in enumerate(gpus_rmvpe):
                    cmd = (
                        '"%s" infer/modules/train/extract/extract_f0_rmvpe.py %s %s %s "%s" %s '
                        % (
                            config.python_cmd,
                            leng,
                            idx,
                            n_g,
                            exp_dir,
                            config.is_half,
                        )
                    )
                    logger.info("Execute: " + cmd)
                    p = Popen(
                        cmd, shell=True, cwd=project_path
                    )  # , shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=now_dir
                    ps.append(p)
                # 煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
                done = [False]
                threading.Thread(
                    target=if_done_multi,  #
                    args=(
                        done,
                        ps,
                    ),
                ).start()
            else:
                cmd = (
                    config.python_cmd
                    + ' infer/modules/train/extract/extract_f0_rmvpe_dml.py "%s" '
                    % (
                        exp_dir,
                    )
                )
                logger.info("Execute: " + cmd)
                p = Popen(
                    cmd, shell=True, cwd=project_path
                )  # , shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=now_dir
                p.wait()
                done = [True]
    while not done[0]: 
        sleep(1)
    # 对不同part分别开多进程
    """
    n_part=int(sys.argv[1])
    i_part=int(sys.argv[2])
    i_gpu=sys.argv[3]
    exp_dir=sys.argv[4]
    os.environ["CUDA_VISIBLE_DEVICES"]=str(i_gpu)
    """
    leng = len(gpus)
    ps = []
    for idx, n_g in enumerate(gpus):
        cmd = (
            '"%s" infer/modules/train/extract_feature_print.py %s %s %s %s "%s" %s %s'
            % (
                config.python_cmd,
                config.device,
                leng,
                idx,
                n_g,
                exp_dir,
                version19,
                config.is_half,
            )
        )
        logger.info("Execute: " + cmd)
        p = Popen(
            cmd, shell=True, cwd=project_path
        )  # , shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=now_dir
        ps.append(p)
    # 煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
    done = [False]
    threading.Thread(
        target=if_done_multi,
        args=(
            done,
            ps,
        ),
    ).start()
    while not done[0]: 
        sleep(1)
    

def arg_parse():
    parser = argparse.ArgumentParser(description="Process GPU configurations and settings for feature extraction.")
    parser.add_argument('--gpus', type=str, default="0", help='GPU devices to use, separated by "-"')
    parser.add_argument('--n_p', type=int, default=8, help='Number of processes to use')
    parser.add_argument('--f0method', type=str, default="rmvpe_gpu", help='Method for F0 extraction')
    parser.add_argument('--if_f0_3', type=bool, default=True, help='Boolean to determine if F0 extraction is needed')
    parser.add_argument('--exp_dir', type=str, default="/home/choi/desktop/rvc/ai/data/user2/output/trained_model", help='Directory for experiment outputs')
    parser.add_argument('--version19', type=str, default="v2", help='Version of the model to use')
    parser.add_argument('--gpus_rmvpe', type=str, default="0-0", help='Specific GPUs for rmvpe method')
    return parser.parse_args()

def main():
    args = arg_parse()
    extract_f0_feature(args.exp_dir, vars(args))

if __name__ == "__main__":
    main()