import json
import os
import re
import shutil
import heapq


class facebook_messages():
    '''
    Use this to delete and clean up your files.
    DELETES group chats and any chats that are not one-on-one chats to save
    space
    '''
    @staticmethod
    def find_close_friends():
        messages_path = os.path.join(os.getcwd(), "messages", "inbox")

        for conversation in os.listdir(messages_path):
            conversation_path = os.path.join(messages_path, conversation)
            try:
                with open(os.path.join(conversation_path, "message.json")) as f:
                    data = json.load(f)
                    if (len(data['messages'])> 30000):
                        print(data['participants'][0]['name'])
                        print(len(data['messages']))
            except NotADirectoryError as e:
                pass
            except Exception as e:
                print(e)

    @staticmethod
    def clean_messages():
        messages_path = os.path.join(os.getcwd(), "messages", "inbox")

        for conversation in os.listdir(messages_path):
            conversation_path = os.path.join(messages_path, conversation)
            try:
                if re.match("facebook", conversation, re.IGNORECASE):
                    # get rid of unknown users
                    shutil.rmtree(conversation_path)
                    continue
                
                with open(os.path.join(conversation_path, "message.json")) as f:
                    data = json.load(f)
                    if data["thread_type"] != "Regular":
                        # get rid of any conversation that's not one-on-one
                        shutil.rmtree(conversation_path)

            except NotADirectoryError as e:
                pass
            except Exception as e:
                print(e)

if __name__ == "__main__":
    facebook_messages.find_close_friends()