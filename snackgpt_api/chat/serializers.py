from rest_framework import serializers
from chat.models import ChatHistory, UserProfile
from django.contrib.auth.models import User
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer


class ChatHistorySerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatHistory
        fields = ['id', 'role', 'message', 'timestamp', 'session_id']
        read_only_fields = ['timestamp', 'role', 'session_id']


class SignupSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ["username", "email", "password"]
        extra_kwargs = {'password': {'write_only': True}}

    def create(self, validated_data):
        user = User.objects.create_user(**validated_data)
        #create_user create user instance and hashes password
        UserProfile.objects.create(user=user)  
        print("user & profile created", user)
        return user
        

class CustomTokenObtainPairSerializer(TokenObtainPairSerializer):
    def validate(self, attrs):
        #attrs is a dict that contains login details (username, password)
        data = super().validate(attrs)
        print("data after login and generating token\n", data)
        print("User instance returned by validate\n", self.user)
        user = self.user
        access_token = data.get("access")
        if access_token:
            profile, _ = UserProfile.objects.get_or_create(user=user)
            profile.jwt_token = access_token
            print("Token Saved..")
            profile.save()
        return data