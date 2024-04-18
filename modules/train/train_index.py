import os
import sys
import logging
import numpy as np
import traceback
import faiss
import platform
import argparse
from sklearn.cluster import MiniBatchKMeans

# Adding current working directory to path
now_dir = os.getcwd()
sys.path.append(now_dir)

from i18n.i18n import I18nAuto
from configs.config import Config

# Setup logging
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Configuration and internationalization
config = Config()
i18n = I18nAuto()

# Environmental variable
outside_index_root = os.getenv("outside_index_root")

def train_index(exp_dir1, version19):
    exp_dir = os.path.join("logs", exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    feature_dir = os.path.join(exp_dir, "3_feature256" if version19 == "v1" else "3_feature768")
    
    if not os.path.exists(feature_dir):
        return "Please perform feature extraction first!"
    
    listdir_res = list(os.listdir(feature_dir))
    if len(listdir_res) == 0:
        return "Please perform feature extraction first!"
    
    infos = []
    npys = []
    for name in sorted(listdir_res):
        path = os.path.join(feature_dir, name)
        phone = np.load(path)
        npys.append(phone)
    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]

    if big_npy.shape[0] > 200000:
        infos.append("Attempting kmeans on array of shape {} to 10k centers.".format(big_npy.shape[0]))
        yield "\n".join(infos)
        try:
            kmeans = MiniBatchKMeans(n_clusters=10000, verbose=True, batch_size=256 * config.n_cpu, compute_labels=False, init="random")
            big_npy = kmeans.fit(big_npy).cluster_centers_
        except Exception as e:
            info = traceback.format_exc()
            logger.info(info)
            infos.append(info)
            yield "\n".join(infos)
    
    np.save(os.path.join(exp_dir, "total_fea.npy"), big_npy)
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    infos.append("{},{}".format(big_npy.shape, n_ivf))
    yield "\n".join(infos)
    
    index_description = "IVF{},Flat".format(n_ivf)
    index = faiss.index_factory(256 if version19 == "v1" else 768, index_description)
    infos.append("training")
    yield "\n".join(infos)
    
    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 1
    index.train(big_npy)
    faiss.write_index(index, os.path.join(exp_dir, "trained_index.index"))
    infos.append("adding")
    yield "\n".join(infos)
    
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i:i + batch_size_add])
    faiss.write_index(index, os.path.join(exp_dir, "trained_index.index"))
    infos.append("Index construction successful: trained_index.index")
    yield "\n".join(infos)

def parse_args():
    parser = argparse.ArgumentParser(description="Train an index with specified directory and version.")
    parser.add_argument("--exp_dir1", type=str, default="../../data/user1/output/trained_model", help="Experiment directory.")
    parser.add_argument("--version19", type=str, default="v2", help="Version specifier.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    for message in train_index(args.exp_dir1, args.version19):
        print(message)

if __name__ == "__main__":
    main()