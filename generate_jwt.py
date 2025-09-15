from livekit_server_sdk import AccessToken

# Dev credentials
API_KEY = "devkey"
API_SECRET = "secret"

# User identity and optional room
identity = "user1"
room_name = "myroom"

# Create the access token
token = AccessToken(API_KEY, API_SECRET, identity=identity, room=room_name)
jwt = token.to_jwt()
print("Dev token:", jwt)
