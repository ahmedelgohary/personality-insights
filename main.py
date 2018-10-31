from flask import Flask
from clean_data import clean_data
app = Flask(__name__)

@app.route('/')
def main():
    clean_data.clean_messages()
    return "Hello, World"
    

if __name__ == "__main__":
    app.run(debug=True)