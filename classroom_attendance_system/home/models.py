from django.db import models
from wagtail.models import Page
from django.utils.timezone import now


class HomePage(Page):
    pass


class User(models.Model):
    name = models.CharField(max_length=50, null=False)
    student_id = models.CharField(max_length=255, unique=True)
    password = models.CharField(max_length=255, null=False)
    is_admin = models.BooleanField(default=False)
    predictor_model = models.BinaryField(null=True, blank=True)
    created_at = models.DateTimeField(default=now)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name


class Attendance(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="attendances")
    created_at = models.DateTimeField(default=now)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Attendance for {self.user.name} on {self.created_at.strftime('%Y-%m-%d')}"
