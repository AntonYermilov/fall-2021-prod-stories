import os
import logging
import deeppavlov
# import python_weather
import re

from telegram import Update
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    Filters,
    CallbackContext
)


def normalize_text(text):
    text = text.lower()
    text = re.sub('[^a-z]', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip()
    return text


class QAModel:
    def __init__(self):
        self.snips = deeppavlov.build_model('intents_snips.json', download=True)
        self.greetings = ["hi", "hello", "hey", "helloo", "hellooo", "g morining", "gmorning", "good morning", "morning", "good day", "good afternoon", "good evening", "greetings", "greeting", "good to see you", "its good seeing you", "how are you", "how're you", "how are you doing", "how ya doin'", "how ya doin", "how is everything", "how is everything going", "how's everything going", "how is you", "how's you", "how are things", "how're things", "how is it going", "how's it going", "how's it goin'", "how's it goin", "how is life been treating you", "how's life been treating you", "how have you been", "how've you been", "what is up", "what's up", "what is cracking", "what's cracking", "what is good", "what's good", "what is happening", "what's happening", "what is new", "what's new", "what is neww", "g’day", "howdy"]
        self.farewells = ["bye", "bye bye", "goodbye", "see you", "see you later", "see you soon", "i'm off", "i am off", "i'm out", "i am out", "take care", "goodnight", "take it easy"]

        self.greetings = list(map(normalize_text, self.greetings))
        self.farewells = list(map(normalize_text, self.farewells))

        # self.weather_client = python_weather.Client(format=python_weather.METRIC)

    def is_greeting(self, text: str):
        return text in self.greetings

    def is_farewell(self, text: str):
        return text in self.farewells

    def __call__(self, text: str):
        text = normalize_text(text)

        if self.is_greeting(text):
            return 'Hi there!'

        if self.is_farewell(text):
            return 'See you!'

        pred, probs = self.snips([text])
        if max(probs[0]) >= 0.45:
            if pred[0] == 'GetWeather':
                # weather = self.weather_client.find("Saint Petersburg")
                return f'2°C, rain'
            elif pred[0] == 'BookRestaurant':
                return 'Sorry, I can\'t book restaurants :('
            elif pred[0] == 'PlayMusic':
                return 'Sorry, I can\'t play music right now :('
            elif pred[0] == 'AddToPlaylist':
                return 'Sorry, I can\'t add songs to playlists right now :('
            elif pred[0] == 'RateBook':
                return 'Sorry, I can\'t rate books right now :('
            elif pred[0] == 'SearchScreeningEvent':
                return 'Sorry, I can\'t search screening events right now :('
            elif pred[0] == 'SearchCreativeWork':
                return 'Sorry, I can\'t search jobs right now :('
        return 'Don\'t understand you :('


qa_model = QAModel()


def on_message(update: Update, context: CallbackContext):
    update.message.reply_text(qa_model(update.message.text))


def on_start(update: Update, context: CallbackContext):
    update.message.reply_text('Привет!')


def init_logger():
    logging.basicConfig(filename='bot.log', filemode='a', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s: %(message)s', datefmt='%Y-%m-%dT%H:%M:%S')


class Bot(object):
    def __init__(self):
        self.updater = Updater(os.environ.get('BOT_TOKEN'))
        self.updater.dispatcher.add_handler(CommandHandler("start", on_start))
        self.updater.dispatcher.add_handler(MessageHandler(Filters.text & (~Filters.command), on_message))
        init_logger()

    def run(self):
        self.updater.start_polling()
        logging.info('Starting bot')
        self.updater.idle()


def main():
    bot = Bot()
    bot.run()


if __name__ == '__main__':
    main()

