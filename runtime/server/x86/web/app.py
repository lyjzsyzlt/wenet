#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2021 Mobvoi Inc. All Rights Reserved.
# Author: zhendong.peng@mobvoi.com (Zhendong Peng)

from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=1234, debug=True)
