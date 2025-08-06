import re
from PIL import Image
from logger import writehistory

def img2txt(pipe, input_text, input_image):
    image = Image.open(input_image)

    writehistory(f"Input text: {input_text} - Type: {type(input_text)} - Dir: {dir(input_text)}")

    if isinstance(input_text, tuple):
        prompt_instructions = (
            "Describe the image using as much detail as possible, is it a painting, a photograph, "
            "what colors are predominant, what is the image about?"
        )
    else:
        prompt_instructions = (
            "Act as an expert in imagery descriptive analysis, using as much detail as possible from the image, "
            "respond to the following prompt: " + input_text
        )

    writehistory(f"prompt_instructions: {prompt_instructions}")
    prompt = "USER: <image>\n" + prompt_instructions + "\nASSISTANT:"

    outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})

    if outputs and outputs[0]["generated_text"]:
        match = re.search(r'ASSISTANT:\s*(.*)', outputs[0]["generated_text"])
        return match.group(1) if match else "No response found."
    return "No response generated."
