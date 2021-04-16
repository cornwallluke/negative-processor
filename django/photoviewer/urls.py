"""photoviewer URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, re_path
import modules.imageendpoint as im
import modules.s3imageendpoint as im2
import modules.controls as ctl
import modules.taskqueue as tq

taskQueue = tq.myQ()

taskQueue.startQueue()

# taskQueue.addAction(lambda : print("hello world"))
# taskQueue.addAction(lambda : print("hello world"))
# taskQueue.addAction(lambda : print("hello world"))

# from time import sleep 
# sleep(2)

# taskQueue.addAction(lambda : print("hello world2"))


# queue = tq.myQ()
# queue = tq.myQ()

# import modules.iprocessors
# im.root = "../photos/"

# im.taskQ = taskQueue



urlpatterns = [
    path('admin/', admin.site.urls),

    re_path(r"old/picture/(?P<album>.+)/(?P<index>\d+)",im.paffedImage),

    re_path(r"old/picture/(?P<album>.+)",im.paffedAlbum),

    re_path(r"old/picture/", im.albums),

    # re_path(r"picture/(?P<album>.+)/(?P<index>\d+)",im.paffedImage),

    # re_path(r"picture/(?P<album>.+)",im.paffedAlbum),

    re_path(r"picture/", im2.albums),

    re_path(r"upload", ctl.upload),

    

]
