from flask import Flask
from facebook_messages import facebook_messages


app = Flask(__name__)


@app.route('/')
def main():
    facebook_messages.find_close_friends()
    return "Hello, World"


@app.route('/clean')
def delete_messages():
    # Get rid of redundant data
    facebook_messages.clean_messages()

    return "Data successfully deleted"


if __name__ == "__main__":
    app.run(debug=True)
