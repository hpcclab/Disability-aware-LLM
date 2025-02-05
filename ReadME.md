# Multimodal Large Language Model (LLM)

A System that is capable of giving audio outputs to guide the user. There is a multimodal language model that processes and interprets both textual and pictorial information. Python and Hugging Face Transformers have been used for its development. Nemoguard rails and LLama Safe Guard modules have been used for safeguarding LLM's output.

## Features

- Process images, text and audio inputs simultaneously.
- Generate audio responses based on user's context.
- Safer responses to the user. Specifically customised to blind and visually imparied persons.
- Provide interactive web interface to the demo the workflow.

## Installation

1. **Clone repository**
```bash
git clone https://github.com/hpcclab/Disability-aware-LLM.git
cd multimodal-llm
```
2. **Run the code**
Run the code in T4 GPU Hardware Accelerator or Higher GPU versions. Used google collab in this project.

3. **Usage of Nemo Guardrails**
Use the configuration files that's in the config directory to use nemogurad rails.


## Project Structure

- *Config.yml:* - Configuration of the nemoguradrails.
- *prompt.yml:* - Predefined prompt templates customised for blind and visually imparied persons.
- *Refactored_LLM_code.ipynb* - Main code of the application.
