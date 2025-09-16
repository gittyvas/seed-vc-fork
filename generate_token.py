from agora_python_server_sdk import AccessToken
import time

# Your Agora credentials
APP_ID = "33e50001273c408fb2f3408415506d75"
APP_CERT = "99c1b87b9daa40e1bb4a1dc5c84f503a"
CHANNEL = "test_channel"
UID = 0  # Use 0 for temporary UID
EXPIRATION = int(time.time()) + 3600  # 1 hour

token = AccessToken(APP_ID, APP_CERT, CHANNEL, UID)
token.add_privilege(AccessToken.PUBLISH_AUDIO, EXPIRATION)
token.add_privilege(AccessToken.PUBLISH_VIDEO, EXPIRATION)

print(token.build())
