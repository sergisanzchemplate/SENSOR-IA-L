from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    ip_addresses = [
        '192.168.50.113',
        '192.168.50.114',
        '192.168.50.115'
    ]
    return render_template('index.html', ip_addresses=ip_addresses)

if __name__ == '__main__':
    app.run(host='192.168.50.129')
