import django.http as djhttp
import urllib
import glob
import os
import json
import modules.iprocessors as iprocessors # pylint: disable=import-error
root = "../photos/"
mediafolders = [
    ["negatives", iprocessors.processnegativedir],
    ["positives", iprocessors.processpositivedir]
]



def list2regx(listy):
    return "("+"|".join(listy)+")"

def paffedImage(req, album, index):
    searchalbum = urllib.parse.unquote_plus(album)
    for mediatype, _ in mediafolders:
        alpath = os.path.join(root, mediatype, searchalbum)
        if os.path.isdir(alpath):
            
            if (os.path.isdir(os.path.join(alpath, "processed")) and 
            os.path.isfile(os.path.join(alpath, "config.json"))):
                if iprocessors.checkHash(alpath):
                    # with  as f:
                    return djhttp.FileResponse(open(sorted(glob.glob(os.path.join(alpath, "processed","*")))[int(index)], "rb"))
    return djhttp.HttpResponseNotFound()
    # return djhttp.FileResponse(open(os.path.join(root, "negatives/Norfolk 2002/monkaOMEGA.jpg"), "rb"))

def paffedAlbum(req, album):#returns config file of album if exists else creates it and starts processing images
    searchalbum = urllib.parse.unquote_plus(album)
    for mediatype, mediahandler in mediafolders:
        alpath = os.path.join(root, mediatype, searchalbum)
        if os.path.isdir(alpath):
            
            if (os.path.isdir(os.path.join(alpath, "processed")) and 
            os.path.isfile(os.path.join(alpath, "config.json"))):
                if iprocessors.checkHash(alpath):
                    with open(os.path.join(alpath, "config.json"), "r") as cfg:
                        return djhttp.HttpResponse(cfg.read())
            #         else:
            #             pass
            #             #make  the things
            # else:
            #     #make the things
            return djhttp.HttpResponse(mediahandler(alpath))
    return djhttp.HttpResponseNotFound()
    # return djhttp.FileResponse(open(os.path.join(root, "negatives/Norfolk 2002/monkaOMEGA.jpg"), "rb"))

def albums(req):
    return djhttp.HttpResponse(json.dumps(list(map(lambda x: os.path.split(x)[-1], glob.glob(os.path.join(root, "*", "*"))))))
