import requests

BASE="http://127.0.0.1:5000/"

response = requests.get(BASE + "helloworld")
response = requests.get(BASE + "users")

print(response.json()) 