import io
import os
import zipfile
from flask import Flask, request, send_file
from openai import OpenAI
from dotenv import load_dotenv
from request_gradio import request_gradio

load_dotenv()
app = Flask(__name__)


@app.route('/call_gradio', methods=['POST'])
def call_gradio():
    if 'file' not in request.files:
        return 'No file part in the request', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    filepath = os.path.join('./', file.filename)
    file.save(filepath)

    client = OpenAI()
    with open(filepath, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format='text'
        )

    # Extract parameters from the request
    seed = int(request.form.get('seed', 0))  # Default is 0
    guidance_scale = int(request.form.get('guidance_scale', 15))  # Default is 15
    inference_steps = int(request.form.get('inference_steps', 64))  # Default is 64

    # Call the request_gradio function
    gradio_response = request_gradio(transcript, seed, guidance_scale, inference_steps)

    bytes_io = io.BytesIO()
    with zipfile.ZipFile(bytes_io, 'w') as zip_f:
        zip_f.writestr('transcript.txt', transcript)

        with open(gradio_response, 'rb') as gradio_file:
            zip_f.writestr(os.path.basename(gradio_response), gradio_file.read())

    bytes_io.seek(0)

    return send_file(bytes_io, as_attachment=True, download_name='archive.zip', mimetype='application/zip')


if __name__ == '__main__':
    app.run()
