# Audo-Sight: Enabling Ambient Interaction for Blind and Visually Impaired Individuals

A System that is capable of giving audio outputs to guide the user. There is a multimodal language model that processes and interprets both textual and pictorial information. Python and Hugging Face Transformers have been used for its development. Nemoguard rails and LLama Safe Guard modules have been used for safeguarding LLM's output.

## Features

- Process images, text and audio inputs simultaneously.
- Generate audio responses based on user's context.
- Safer responses to the user. Specifically customised to blind and visually imparied persons.
- Provide interactive web interface to the demo the workflow.

## Installation & Run
This project has not been dockerized yet. You should install it on your own. It was implemented with Python 3.8 and has been tested on Ubuntu 22.04. But, the lower version of them would be fine.

1. Install the required Python package.
```bash
pip3 install -r requirements.txt
```

## Project Structure

- *Config.yml:* - Configuration of the nemoguradrails.
- *prompt.yml:* - Predefined prompt templates customised for blind and visually imparied persons.
- *Refactored_LLM_code.ipynb* - Main code of the application.
