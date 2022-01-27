# import torch
# import requests
#
# CLIENT_ID = ''
# CLIENT_SECRET = ''
#
#
# AUTH_URL = 'https://accounts.spotify.com/api/token'
#
# # POST
# auth_response = requests.post(AUTH_URL, {
#     'grant_type': 'client_credentials',
#     'client_id': CLIENT_ID,
#     'client_secret': CLIENT_SECRET,
# })
#
# # convert the response to JSON
# auth_response_data = auth_response.json()
#
# # save the access token
# access_token = auth_response_data['access_token']
#
# headers = {
#     'Authorization': 'Bearer {token}'.format(token=access_token)
# }
#
# # base URL of all Spotify API endpoints
# BASE_URL = 'https://api.spotify.com/v1/'
#
# # Track ID from the URI
# track_id = '6y0igZArWVi6Iz0rj35c1Y'
#
# # actual GET request with proper header
# r = requests.get(BASE_URL + 'audio-features/' + track_id, headers=headers)
#
# r = r.json()
# print(r)

import matplotlib.pyplot as plt
from facelib import AgeGenderEstimator, FaceDetector, EmotionDetector


img = plt.imread('download.jpeg')
face_detector = FaceDetector()
face_detector_emotion = FaceDetector(face_size=(224, 224))
age_gender_detector = AgeGenderEstimator()
emotion_detector = EmotionDetector()

faces, boxes, scores, landmarks = face_detector.detect_align(img)
genders, ages = age_gender_detector.detect(faces)
print(genders, ages)


faces, boxes, scores, landmarks = face_detector_emotion.detect_align(img)
list_of_emotions, probab = emotion_detector.detect_emotion(faces)
print(list_of_emotions)
