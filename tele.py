#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 14:34:25 2018

@author: igor
"""


from typing import Any, Callable
from telegram import Bot, Update
from telegram.ext import Updater
from telegram.ext import CommandHandler
import logging
import secrets as s
import qlib as q

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger()

def authorized_only(command_handler: Callable[[Any, Bot, Update], None]) -> Callable[..., Any]:
    """
    Decorator to check if the message comes from the correct chat_id
    :param command_handler: Telegram CommandHandler
    :return: decorated function
    """
    def wrapper(*args, **kwargs):
        """ Decorator logic """
        update = kwargs.get('update') or args[1]

        if int(update.message.chat_id) != s.telegram_chat_id:
            logger.info(
                'Rejected unauthorized message from: %s',
                update.message.chat_id
            )
            return wrapper

        logger.info(
            'Executing handler: %s for chat_id: %s',
            command_handler.__name__,
            s.telegram_chat_id
        )
        try:
            return command_handler(*args, **kwargs)
        except BaseException:
            logger.exception('Exception occurred within Telegram module')

    return wrapper

@authorized_only
def status(bot: Bot, update: Update) -> None:
    send_msg('Signal: ' + q.get_signal())

updater = Updater(token=s.telegram_token, workers=0)
updater.dispatcher.add_handler(CommandHandler('status', status))
updater.start_polling(clean=True, bootstrap_retries=-1, timeout=300, read_latency=60)

def send_msg(msg):
    updater.bot.send_message(chat_id=s.telegram_chat_id, text=msg)
    updater.bot.send_message(chat_id=716893261, text=msg)

def cleanup():
    updater.stop()
