from flask import Flask, request, send_file
from request_gradio import request_gradio

app = Flask(__name__)


@app.route('/call_gradio', methods=['POST'])
def call_gradio():
    # Extract parameters from the request
    data = request.json
    prompt = data.get('prompt')
    seed = data.get('seed', 0)  # Default value is 0
    guidance_scale = data.get('guidance_scale', 15)  # Default value is 15
    inference_steps = data.get('inference_steps', 64)  # Default value is 64
    url = data.get('url', "https://hysts-shap-e.hf.space/--replicas/vxg55/")  # Default value
    api_name = data.get('api_name', "/text-to-3d")  # Default value

    # Call the request_gradio function
    response = request_gradio(prompt, seed, guidance_scale, inference_steps, url, api_name)

    # Return the response
    return send_file(response, as_attachment=True)


if __name__ == '__main__':
    app.run()
