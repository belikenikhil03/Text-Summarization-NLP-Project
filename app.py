from flask import Flask, request, jsonify
from transformers import pipeline
from transformers import pipeline
from flask_cors import CORS


app = Flask(__name__)
