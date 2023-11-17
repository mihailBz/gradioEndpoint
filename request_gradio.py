from gradio_client import Client


def request_gradio(
        prompt,
        seed=0,
        guidance_scale=15,
        inference_steps=64,
        url="https://hysts-shap-e.hf.space/--replicas/fwz55/",
        api_name="/text-to-3d"
):
    return Client(url).predict(
        prompt,  # str  in 'Prompt' Textbox component
        seed,  # float (numeric value between 0 and 2147483647) in 'Seed' Slider component
        guidance_scale,  # float (numeric value between 1 and 20) in 'Guidance scale' Slider component
        inference_steps,  # float (numeric value between 2 and 100) in 'Number of inference steps' Slider component
        api_name=api_name
    )
