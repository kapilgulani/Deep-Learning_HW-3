from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import torch
from transformers import pipeline
from DeepDataMiningLearning.hfaudio.inference import MyAudioInference

# Load models
translate_pipeline = pipeline("translation_en_to_hi", model="Helsinki-NLP/opus-mt-en-zh")
summarize_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
qa_model = pipeline('question-answering', model='deepset/roberta-base-squad2')

def summarize_text(text):
    return summarize_pipeline(text)[0]['summary_text']

app = Flask(__name__)
CORS(app)

# Configurations
app.config['UPLOAD_FOLDER'] = 'uploaded_files'
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def is_allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/file-upload', methods=['POST'])
def handle_file_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and is_allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Perform model inference
        model_name = "facebook/wav2vec2-large-robust-ft-libri-960h"
        task = "audio-asr"
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        cache_dir = "./output_cache"

        audio_infer = MyAudioInference(model_name, task=task, target_language='eng', cache_dir=cache_dir)
        result = audio_infer(file_path)

        os.remove(file_path)  # Clean up after processing
        return jsonify({'result': result})

    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/text-translate', methods=['POST'])
def translate():
    data = request.get_json()
    text = data.get('text')
    result = translate_pipeline(text)
    return jsonify({'result': result[0]['translation_text']})

@app.route('/text-answer', methods=['POST'])
def answer():
    data = request.get_json()
    context = data.get('text')
    question = data.get('question')
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    if not context:
        return jsonify({'error': 'No context provided'}), 400
    answer = qa_model(question=question, context=context)
    return jsonify({'result': answer['answer']})

@app.route('/text-summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    text = data.get('text')
    summary = summarize_text(text)
    return jsonify({'result': summary})

if __name__ == '__main__':
    app.run(debug=True)
