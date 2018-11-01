#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 14:34:25 2018

@author: igor
"""

import secrets as s

from telegram.ext import Updater
from telegram.ext import CommandHandler
import logging

def start(bot, update):
    bot.send_message(chat_id=update.message.chat_id, text="I'm a bot, please talk to me!")

updater = Updater(token=s.telegram_token)
dispatcher = updater.dispatcher
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)


start_handler = CommandHandler('start', start)
dispatcher.add_handler(start_handler)

updater.start_polling()

def send_msg(msg):
    updater.bot.send_message(chat_id=s.telegram_chat_id, text=msg)
