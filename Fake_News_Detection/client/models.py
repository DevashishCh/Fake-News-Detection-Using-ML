from django.db import models

# Create your models here.

class clientinformation(models.Model):
    id = models.AutoField(primary_key=True)
    firstname=models.CharField(max_length=300)
    lastname=models.CharField(max_length=200)
    userid=models.CharField(max_length=200)
    password=models.CharField(max_length=200)
    phoneno=models.BigIntegerField()
    email=models.CharField(max_length=200)
    gender=models.CharField(max_length=200)


class FakeRealModel(models.Model):
    fakenews=models.CharField(max_length=100)
    realnews=models.CharField(max_length=100)
    alpha=models.CharField(max_length=100)



class AccuracyModel(models.Model):
    accuracy=models.CharField(max_length=100)

class AccuracyModel1(models.Model):
    accuracy1=models.CharField(max_length=100)


class AccuracyModel2(models.Model):
    accuracy2=models.CharField(max_length=100)



class AccuracyModel3(models.Model):
    accuracy3=models.CharField(max_length=100)

