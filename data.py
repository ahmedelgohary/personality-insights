import json, os, re, shutil

class data():
    @staticmethod
    def clean_messages():
        os.chdir(str(os.getcwd()) + "/messages")

        for conversation in os.listdir():
            try:
                if re.match("facebook", conversation, re.IGNORECASE):   
                    shutil.rmtree(os.getcwd() + "/" + conversation)   # get rid of unknown users
                with open(conversation + "/message.json") as f: 
                    data = json.load(f)
                    if data["thread_type"] != "Regular":
                        shutil.rmtree(os.getcwd() + "/" + conversation)   # get rid of any conversation that's not one-on-one
                    
            except NotADirectoryError:
                pass
            except FileNotFoundError:
                shutil.rmtree(os.getcwd() + "/" + conversation)

