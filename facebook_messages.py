import json
import os
import re
import shutil


class facebook_messages():
    '''
    Use this to delete and clean up your files. 
    DELETES group chats and any chats that are not one-on-one chats to save 
    space
    '''
    @staticmethod
    def find_close_friends():
        for conversation in os.listdir(os.path.join(os.getcwd(), "messages")):
            try:
                with open(conversation + "/message.json") as f:
                    data = json.load(f)
                    if data["thread_type"] == "Regular":
                        print(data["title"], len(data['messages']))

            except NotADirectoryError:
                pass
            except FileNotFoundError:
                # get rid of useless folders
                print("File not found")

    @staticmethod
    def clean_messages():
        os.chdir(os.path.join(os.getcwd(), "/messages"))

        for conversation in os.listdir():
            try:
                if re.match("facebook", conversation, re.IGNORECASE):
                    # get rid of unknown users
                    shutil.rmtree(os.getcwd() + "/" + conversation)
                with open(conversation + "/message.json") as f:
                    data = json.load(f)
                    if data["thread_type"] != "Regular":
                        # get rid of any conversation that's not one-on-one
                        shutil.rmtree(os.getcwd() + "/" + conversation)

            except NotADirectoryError:
                pass
            except FileNotFoundError:
                # get rid of useless folders
                shutil.rmtree(os.getcwd() + "/" + conversation)
