import random

from gradio_client import Client


def request_gradio(
        prompt,
        seed=random.randint(0, 2147483647),
        guidance_scale=15,
        inference_steps=64,
        url="https://hysts-shap-e.hf.space/",
        api_name="/text-to-3d"
):
    return Client(url, verbose=False).predict(
        prompt,  # str  in 'Prompt' Textbox component
        random.randint(0, 2147483647),  # float (numeric value between 0 and 2147483647) in 'Seed' Slider component
        guidance_scale,  # float (numeric value between 1 and 20) in 'Guidance scale' Slider component
        inference_steps,  # float (numeric value between 2 and 100) in 'Number of inference steps' Slider component
        api_name=api_name
    )
