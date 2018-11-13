import json, os, re, shutil

class clean_data():
    @staticmethod
    def clean_messages():
        os.chdir(str(os.getcwd()) + "/messages")

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

