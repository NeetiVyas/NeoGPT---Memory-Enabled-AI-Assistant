import uuid
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.decorators import api_view, permission_classes
from django.db.models import Q
from rest_framework.response import Response
from django.contrib.auth.decorators import login_required
from rest_framework.permissions import IsAuthenticated, AllowAny
from chat.models import ChatHistory
from rest_framework import status
from rest_framework_simplejwt.views import TokenObtainPairView
from django.contrib.auth import authenticate
from chat.serializers import ChatHistorySerializer, SignupSerializer, CustomTokenObtainPairSerializer


class SignupView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        #performing deserialization
        serializer = SignupSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            return Response({"msg": "signup successfull"}, status=status.HTTP_201_CREATED)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


'''class LoginView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        username = request.data.get("username")
        password = request.data.get("password")
        user = authenticate(username=username, password=password)
        if user is None:
            Response({"error": "Invalid credentials"}, status=status.HTTP_401_UNAUTHORIZED)'''
    

class CustomTokenObtainPairView(TokenObtainPairView):
    print("Login view getting called\n")
    serializer_class = CustomTokenObtainPairSerializer
        

class ChatHistoryView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        session_id = request.query_params.get("session_id")
        print("redddnjasjasbfasdygvghdh",request.user)
        if session_id:
            qs = ChatHistory.objects.filter(user=request.user, session_id=session_id).order_by('timestamp')
        else:
            qs = ChatHistory.objects.filter(user=request.user).order_by('timestamp')
        serializer = ChatHistorySerializer(qs, many=True)
        print("Serializer data get api ", serializer.data)
        return Response(serializer.data)

    def post(self, request):
        user_message = request.data.get("message")
        assistant_message = request.data.get("assistant_message")
        session_id = request.data.get("session_id")
        if not user_message or not assistant_message:
            return Response({"error": "message and assistant_message are required"}, status=400)
        
        if not session_id:
            session_id = uuid.uuid4()
        
        # Save user message
        print("user msg & bot msg received")
        user_msg = ChatHistory.objects.create(
            user=request.user,
            role='user',
            message=user_message,
            session_id = session_id
        )

        assistant_msg = ChatHistory.objects.create(
            user=request.user,
            message=assistant_message,
            role='assistant',
            session_id=session_id
        )

        return Response({
            "user_message": ChatHistorySerializer(user_msg).data,
            "assistant_message": ChatHistorySerializer(assistant_msg).data,
            "session_id": session_id
        }, status=status.HTTP_201_CREATED)
    
    def delete(self, request):
        session_id = request.query_params.get("session_id")
        if not session_id:
            return Response({"error": "session_id is required"}, status=status.HTTP_400_BAD_REQUEST)
        
        deleted_count, _ = ChatHistory.objects.filter(user=request.user, session_id=session_id).delete()

        if deleted_count==0:
            return Response({"error": "No messages found for this session"}, status=status.HTTP_404_NOT_FOUND)
        else:
            return Response({"message": f"Deleted {deleted_count} messages for session {session_id}"}, status=status.HTTP_200_OK)
        

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def search_chat(request):
    query = request.GET.get("q", "").strip()
    if not query:
        return Response({"Error": "Query parameter is required"}, status=400)
    user=request.user
    results = ChatHistory.objects.filter(
        user=user,
        message__icontains = query
        ).order_by("-timestamp")[:20]
    
    data = ChatHistorySerializer(results, many=True).data
    return Response(data, status=200)



