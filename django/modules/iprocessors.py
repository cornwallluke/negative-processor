import rawpy
import numpy as np
from scipy.ndimage.filters import convolve
import glob
import os
import json
import imageio
from cv2 import cv2
# H_G = np.asarray(
#     [[0, 1, 0],
#     [1, 4, 1],
#     [0, 1, 0]], dtype=np.float64 ) / 4

# H_RB = np.asarray(
#     [[1, 2, 1],
#     [2, 4, 2],
#     [1, 2, 1]], dtype=np.float64) / 4
DEFAULTGAMMA = 2.2
DEFAULTCONF = {
        "saturation":0,
        "whitebalance":0,
        "hash":0,
        "images":0,
    }


def rawToMat(ppath):
    with rawpy.imread(ppath) as raw:
        # rpyraws.append(raw)
        imcopy = raw.raw_image_visible.copy()
        pog = cv2.demosaicing(raw.raw_image_visible, cv2.COLOR_BayerRG2BGR)
        pog = pog/pog.max()
        pog = pog ** (1/DEFAULTGAMMA)
        # pog = np.concatenate((
        #     convolve((imcopy*(raw.raw_colors_visible==0)), H_RB)[:,:,np.newaxis],
        #     convolve((imcopy*(raw.raw_colors_visible%2)), H_G)[:,:,np.newaxis],
        #     convolve((imcopy*(raw.raw_colors_visible==2)), H_RB)[:,:,np.newaxis],
        # ), axis=2) ** (1/DEFAULTGAMMA)
        return pog
def dirsize(direct):
    cum = 0
    for _,_, files in os.walk(direct):
        for f in files:
            cum+=os.path.getsize(os.path.join(direct, f))
    return cum 
def checkHash(direct):
    if os.path.isfile(os.path.join(direct, "config.json")):
        with open(os.path.join(direct, "config.json"), "r") as cfg:
            info = json.loads(cfg.read())
            print(info)
            print(dirsize(os.path.join(os.path.join(direct, "processed"))))
            return "hash" in info and info["hash"] == dirsize(os.path.join(os.path.join(direct, "processed")))
    return False

def process(direct, wpfunc, bpfunc, conf):
    raws = []
    print(f"processing {direct}")
    if checkHash(direct):
        print("already Processed directory sent for processing")
    for i in sorted(glob.glob(os.path.join(direct,"*.NEF"))):

        print(f"loading {i} for whitebalance in {direct}")
        rmat = rawToMat(i)
        raws.append(rmat[500:-500:20, 500:-500:20])
    raws=np.asarray(raws)

    conf["images"] = len(raws)
    print("raws loaded")
    pogawbb = bpfunc(raws) #np.percentile(raws[:,::10,::10], 10, (0, 1, 2))
    pogawbw = wpfunc(raws) #np.percentile(raws[:,::10,::10], 99, (0, 1, 2))
    print("wb calc")
    # raws = (raws-pogawbb)/(pogawbw-pogawbb)
    # print("wb'd")
    if not os.path.isdir(os.path.join(direct, "processed")):
        os.mkdir(os.path.join(direct, "processed"))
    ppaths = sorted(glob.glob(os.path.join(direct,"*.NEF")))
    for i in range(len(raws)):
        rawmat = rawToMat(ppaths[i])
        wbd = (rawmat-pogawbb)/(pogawbw-pogawbb)
        sat = 1.5+.01*conf["saturation"]

        contrasted = 0.2 * wbd + 0.8 * ( wbd - wbd[300:-300, 300:-300].min() )/(wbd[300:-300, 300:-300].max() - wbd[300:-300, 300:-300].min())
        
        # brightnessed = contrasted + wbd.mean() * 0.1

        towrite = cv2.cvtColor(
            (
                cv2.cvtColor(
                    contrasted.clip(0,1).astype(np.float32),
                    cv2.COLOR_BGR2HSV
                ) * [1, sat, 1]
            ).astype(np.float32),
            cv2.COLOR_HSV2BGR
        )
        # towrite = matplotlib.colors.hsv_to_rgb(matplotlib.colors.rgb_to_hsv(contrasted)*[1, sat, 1])
        print(f"saturated raw {i}")
    
        imageio.imwrite(os.path.join(direct, "processed", f"generated{i:04}.jpg"), (towrite.clip(0, 1)*255).astype(np.uint8))
    
    conf["hash"] = dirsize(os.path.join(direct, "processed"))

    with open(os.path.join(direct,"config.json"), "w") as cfg:
        cfg.write(json.dumps(conf))
    
    return conf

    


def processdir(direct, wpfunc, bpfunc, taskQ):
    #given a directory make a config file and 
    
    # rpyraws = []
    conf = DEFAULTCONF.copy()
    if os.path.isfile(os.path.join(direct, "config.json")):
        with open(os.path.join(direct,"config.json"), "w") as cfg:
            loaded = json.loads(cfg.read())
            for key in conf:
                conf[key] = key in loaded and loaded[key] or conf[key]

    
    taskQ.addAction(lambda : process(direct, wpfunc, bpfunc, conf))
    
    conf["images"] = len(glob.glob(os.path.join(direct,"*.NEF")))
    return json.dumps(conf)

def processnegativedir(direct, taskQ):
    return processdir(
        direct, 
        lambda raws:np.percentile(raws, 10, (0, 1, 2)),
        lambda raws:np.percentile(raws, 99, (0, 1, 2)),
        taskQ
    )

def processpositivedir(direct, taskQ):
    return processdir(
        direct, 
        lambda raws:np.percentile(raws, 99, (0, 1, 2)),
        lambda raws:np.percentile(raws, 1, (0, 1, 2)),
        taskQ
    )
    
