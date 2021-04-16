import django.http as djhttp
import boto3
import urllib
import glob
import os
import json
import modules.iprocessors as iprocessors # pylint: disable=import-error


dbimages = boto3.resource("dynamodb").Table("images")
dbalbums = boto3.resource("dynamodb").Table("albums")

buckett = boto3.resource("s3").Bucket("cornwalljonesphotobucket")

taskQ = None


# def list2regx(listy):
#     return "("+"|".join(listy)+")"

def paffedImage(req, album, index):
    imid = dbalbums.get_item(Key={"id":"overyalbumtest"})["Item"]["images"][index]
    iminfo = dbimages.get_item(Key={"id":imid})["Item"]
    print(iminfo)
    if !iminfo["processed"]:
        #process the image
        pass
    return 
#     searchalbum = urllib.parse.unquote_plus(album)
#     for mediatype, _ in mediafolders:
#         alpath = os.path.join(root, mediatype, searchalbum)
#         if os.path.isdir(alpath):
            
#             if (os.path.isdir(os.path.join(alpath, "processed")) and 
#             os.path.isfile(os.path.join(alpath, "config.json"))):
#                 if iprocessors.checkHash(alpath):
#                     # with  as f:
#                     return djhttp.FileResponse(open(sorted(glob.glob(os.path.join(alpath, "processed","*")))[int(index)], "rb"))
#     return djhttp.HttpResponseNotFound()
#     # return djhttp.FileResponse(open(os.path.join(root, "negatives/Norfolk 2002/monkaOMEGA.jpg"), "rb"))

def paffedAlbum(req, album):#returns config file of album if exists else creates it and starts processing images
    djhttp.HttpResponse(json.dumps({"images": len(dbalbums.get_item(Key={"id":"overyalbumtest"})["Item"]["images"])}))
#     searchalbum = urllib.parse.unquote_plus(album)
#     for mediatype, mediahandler in mediafolders:
#         alpath = os.path.join(root, mediatype, searchalbum)
#         if os.path.isdir(alpath):
            
#             if (os.path.isdir(os.path.join(alpath, "processed")) and 
#             os.path.isfile(os.path.join(alpath, "config.json"))):
#                 if iprocessors.checkHash(alpath):
#                     with open(os.path.join(alpath, "config.json"), "r") as cfg:
#                         return djhttp.HttpResponse(cfg.read())
#             #         else:
#             #             pass
#             #             #make  the things
#             # else:
#             #     #make the things
#             return djhttp.HttpResponse(mediahandler(alpath, taskQ))
#     return djhttp.HttpResponseNotFound()
#     # return djhttp.FileResponse(open(os.path.join(root, "negatives/Norfolk 2002/monkaOMEGA.jpg"), "rb"))

def albums(req):
    return djhttp.HttpResponse(json.dumps(list(map(lambda x: x["id"], dbalbums.scan(Select="SPECIFIC_ATTRIBUTES", AttributesToGet=["id"])["Items"]))))
