import django.http as djhttp
def hello(req):
    return djhttp.HttpResponse("hello world")
def imglmao(req):
    return djhttp.FileResponse(open("../../../Pictures/8gdump/monkaOMEGA.jpg", "rb"))