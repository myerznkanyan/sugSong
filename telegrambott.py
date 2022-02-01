import os
import telebot
import matplotlib.pyplot as plt
from pathlib import Path
from mood import doit
from age_gender_ethnicity import gender_detection, age_detection, ethnicity_detection
print(222222)

API_KEY = os.getenv("5124948630:AAFI8yFW6-gyGHdcOo-R777lLEMuen0EKAw")  # mood_classify
bot = telebot.TeleBot('5124948630:AAFI8yFW6-gyGHdcOo-R777lLEMuen0EKAw')  # mood_classify_bot

print("lvoe")
@bot.message_handler(commands=["hello"])
def greet(message):
    bot.send_message(message.chat.id, 'Hi there! I am AIBOT,just send me your photo\
                                      and you will see the magic of AI')
    @bot.message_handler(content_types=['photo'])
    def photo(message):
        print('message.photo =', message.photo)
        fileID = message.photo[-1].file_id
        print('fileID =', fileID)
        file_info = bot.get_file(fileID)
        print('file.file_path =', file_info.file_path)
        downloaded_file = bot.download_file(file_info.file_path)

        with open("image.jpg", 'wb') as new_file:
            new_file.write(downloaded_file)
        bot.reply_to(message, "wait for a second !!!")

        image_url = None
        image_formats = ("image.png", "image.jpeg", "image.jpg")
        if path == Path("image.jpg"):
            c= None

            new_image = plt.imread("image.jpg")
            print("="*40)
            img = plt.imshow(new_image)
            mood = doit(new_image)
            age = age_detection(new_image)
            gender = gender_detection(new_image)

            ethnicity =ethnicity_detection(new_image)
            if mood=="happy" and gender=="male":
                bot.reply_to(message, f"Great! I wish to see you always happy bro!! Do you want to countiue with me ?")
                bot.reply_to(f"Soo as my model said to me you are {age} years old and your ethnicity is {ethnicity}:\
                        as I am begginer maybe you are years older or younger, Now that is all that I can do for you, was that good?")
                bot.reply_to("thank you very much bro!!!")
            if  mood== "angry" or "disgust" or "fear" or "netural" and gender == "male":
                bot.reply_to(message, f"I got that you are {mood}, unforutantly I cannot help\
                                         you yet. I hope one day I will.\
                                         Do you want to countiue with me ?")
                bot.reply_to(f"Soo as my model said to me you are {age} years old and your ethnicity is {ethnicity}:\
                                 as I am begginer maybe you are years older or younger,\
                                Now that is all that I can do for you, was that good?")
                bot.reply_to("thank you very much bro!!!")

            return c
        return "havent send photo"
    print("2222222"*70)



path = Path("image.jpg")
bot.polling()

