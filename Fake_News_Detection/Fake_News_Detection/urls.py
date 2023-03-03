"""Fake_News_Detection URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls.static import static
from django.contrib import admin
from django.conf.urls import url

from Fake_News_Detection import settings
from client import views as user_view
from management import views as admin_view

urlpatterns = [
    url(r'^admin/', admin.site.urls),


    url(r'^$', user_view.login, name="login"),
    url(r'^register/$', user_view.register, name="register"),
    url(r'^mydetails/$', user_view.mydetails, name="mydetails"),
    url(r'^uploadpage/$', user_view.uploadpage, name="uploadpage"),
    url(r'^analysis/$', user_view.analysis, name="analysis"),
    url(r'^analysis1/$', user_view.analysis1, name="analysis1"),
    url(r'^analysis2/$', user_view.analysis2, name="analysis2"),
    url(r'^analysis3/$', user_view.analysis3, name="analysis3"),
    url(r'^analysis31/$', user_view.analysis31, name="analysis31"),
    url(r'^analysis32/$', user_view.analysis32, name="analysis32"),
    url(r'^analysis4/$', user_view.analysis4, name="analysis4"),
    url(r'^analysis5/$', user_view.analysis5, name="analysis5"),
    url(r'^view_uploadnews/$', user_view.view_uploadnews, name="view_uploadnews"),


    url(r'^loginpage/$', admin_view.loginpage, name="loginpage"),

    url(r'^upload_dataset/$', admin_view.upload_dataset, name="upload_dataset"),
    url(r'^uploadpage1/$', admin_view.uploadpage1, name="uploadpage1"),


]+static(settings.MEDIA_URL, document_root= settings.MEDIA_ROOT)
