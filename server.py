import logging
from flask import Flask, request, jsonify
import dill
from bs4 import BeautifulSoup
import requests
import json

from extractor.content_extractor import extract
from extractor.parser import Parser
from extractor.util import load_log_config

logging.config.dictConfig(load_log_config())
logger = logging.getLogger('applog.' + __name__)

app = Flask(__name__)

with open('data/model.pkl', 'rb') as f:
    model = dill.load(f)
    logger.info('loaded model')

@app.route('/extract/body', methods=['POST'])
def extract_content():
    result = {}
    try:
        params = get_params(request.json)
        html = params['html']

        r = extract(model, html, True)
        return {
            'content':    r['content'],
            'image_urls': r['image_urls'],
            'score':      r['score'],
        }

    except Exception as e:
        result['status'] = 'NG'
        result['error'] = repr(e)
        logger.error(repr(e))
        return jsonify(result)

    result['status'] = 'OK'
    return jsonify(result)

@app.route('/test', methods=['GET'])
def test_extract_content():
    url = request.args.get('url')
    res = requests.get(url)
    res.encoding = res.apparent_encoding
    data = json.dumps({'html': res.text})
    r = requests.post('http://app:5000/extract/body', data=data, headers={'content-type': 'application/json'})
    return r.text

@app.route('/healthcheck', methods=['GET'])
def ping():
    return 'OK'

def get_params(data):
    params = {}
    if 'html' not in data:
        raise Exception('html is required')
    params['html'] = data['html']

    return params
