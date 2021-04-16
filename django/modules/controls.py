import django.http as djhttp
import hashlib
import boto3
from django.views.decorators.csrf import csrf_exempt

from cv2 import cv2
import numpy as np
import rawpy




dbimages = boto3.resource("dynamodb").Table("images")
dbalbums = boto3.resource("dynamodb").Table("albums")

buckett = boto3.resource("s3").Bucket("cornwalljonesphotobucket")


@csrf_exempt
def upload(req):
    #create empty album with albumname
    neg = req.POST["negative"]
    album = {
        "id":req.POST["album"],
        "images":[],
        "whitepoint":[ 0 for i in range(3)],
        "blackpoint":[ 1 for i in range(3)],
        "saturation":1,
        "display":True,
        "negative":neg
    }
    


    for i in req.FILES:
        buf = req.FILES[i]
        if not hasattr(buf, "temporary_file_path"):
            return djhttp.HttpResponse("fuck", status=500)
            # imid = hashlib.sha384(buf.read())
            
            # rpy = rawpy.RawPy() # pylint: disable=no-member
            
            # rpy.open_buffer(req.FILES[i])
            # rpy.unpack()
        with open(buf.temporary_file_path(), "rb") as f:
            imid = hashlib.sha384(f.read()).hexdigest()

        
        rpy = rawpy.imread(buf.temporary_file_path())
            

        
        # check image by file hash not in ddb

        #add to cumulative album summary data
        pp = rpy.postprocess(output_bps=16, user_wb=[1, 1, 1, 1], gamma=(1, 1), no_auto_bright=True, user_flip=0)
        maxval = pp[300:-300:10, 300:-300:10].max()  
        album["blackpoint"] = list(np.minimum(np.percentile(pp[300:-300:10, 300:-300:10], 0.01, axis=(0, 1)) / maxval, album["blackpoint"]))
        album["whitepoint"] = list(np.maximum(np.percentile(pp[300:-300:10, 300:-300:10], 99.9, axis=(0, 1)) / maxval, album["whitepoint"]))

        album["images"].append(imid)

        #save raw in s3 sha384+"raw"

        buckett.upload_file(buf.temporary_file_path(), imid+"raw")
        #save image in ddb with empty processed version
        dbimages.put_item(Item={
            "id":imid,
            "comments":[],
            "blackpoint":[0 for i in range(3)],
            "whitepoint":[0 for i in range(3)],
            "processed":False
        })
        #add hash to album image list

        
        
        # with open("test"+i+".jpg", "wb+") as wtable:
        print(hashlib.sha256(req.FILES[i].read()))
    album["whitepoint"] = list(map(lambda x: str(x), album["whitepoint"]))
    album["blackpoint"] = list(map(lambda x: str(x), album["blackpoint"]))
    dbalbums.put_item(Item=album)

    return djhttp.JsonResponse({})