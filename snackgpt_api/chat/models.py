import uuid
from django.db import models
from django.contrib.auth.models import User


class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    jwt_token = models.CharField(max_length=512, blank=True, null=True)

class ChatHistory(models.Model):

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    role = models.CharField(max_length=10)
    message = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    session_id = models.UUIDField(default=uuid.uuid4, null=True, blank=True)

    def __str__(self):
        return f"{self.user.username} --> {self.role}"