from flask import Flask, request

from main import chat_tfidf

app = Flask(__name__)


@app.route('/chatbot', methods=['GET'])
def hello_world():
    return chat_tfidf(request.args.get('text'))


if __name__ == '__main__':
    app.run('localhost', 5000)
