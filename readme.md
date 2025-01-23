# Temperature_eval


This repository serves as a tool to identify the optimal temperature settings for a given prompt and model, leveraging findings from the following paper:

**Exploring the Impact of Temperature on Large Language Models: Hot or Cold?**

<p align="center">
  <img src="assets/images/head.jpeg" alt="Empirical Analysis" width="300">
</p>


This tool provides a structured guide for usage, starting with the **Prompt Input Section**, where users input a **prompt** and select a **model** for evaluation. Afterward, clicking the **"Analyze"** button initiates the process, utilizing the **[Volavion/bert-base-multilingual-uncased-temperature-cls](https://huggingface.co/Volavion/bert-base-multilingual-uncased-temperature-cls)** model, which is **pre-trained** and **fine-tuned** for temperature classification without requiring special permissions. The output includes a **bar chart** visualizing class label probabilities and a **"Best Temperature"** text box showing the **optimal temperature** derived from the model's performance distribution. Users can fine-tune this value via a **sliding bar**. For generation tasks, an **API key** in either the **vLLM OpenAI API** or **Ollama API** format is required. If a **CSV file** with a column named **"input"** is uploaded, the tool appends a **"generated response"** column to the output file, which can be downloaded easily.

<!-- 
## Abilities

This tool is designed to classify six key abilities of large language models (LLMs): **Causal Reasoning (CR)**, the ability to derive conclusions based on logical principles; **Creativity (CT)**, involving the generation of novel and effective ideas; **Instruction Following (IF)**, reflecting adherence to prompts and guidelines; **In-Context Learning (ICL)**, showcasing the capacity to learn and perform tasks using contextual examples; **Summarization (SUM)**, which condenses lengthy texts while preserving key points; and **Machine Translation (MT)**, enabling accurate translation between languages. Based on these classifications, the tool identifies and reports the best temperature for optimal performance. -->


## How to use

Follow the steps below to configure the environment and install the required dependencies for the tool:

1. **Create and Activate the Conda Environment**  
   Execute the following commands to set up a dedicated Conda environment:  
   ```bash
   conda create -n env_temperature_eval python=3.10
   conda activate env_temperature_eval
   ```

2. **Install Dependencies**  
   Install the necessary Python libraries using `pip`:  
   ```bash
   pip install gradio transformers torch matplotlib
   ```
3. **(Optional) Set Up Your API server**

   Considering the API calling and format is different, we rovide an interface to call the API by allowing you to set the curl API format.

   When you use vllm server，

   When you use ollama server，

   When you use you own API，

   When you use openAI API，

   
4. **Start the Tool**  
   
   Launch the server by running the main script:  
   ```bash
   python main_start.py
   ```

   After the server starts, a link will appear in the terminal. Click on this link to access and interact with the tool. Follow the steps below depending on your desired operation:

   1. **Single Prompt Temperature Selection**:  
      - Input your prompt directly into the tool.  
      - Select the most suitable model based on the **model selection guidelines** provided.  
      - The tool will calculate and display the best temperature for your prompt.

   2. **Batch Prompt Temperature Selection**:  
      - Upload a CSV file containing your prompts.  
      - Ensure the file has at least one column named **"input"** to represent your prompts.  
      - The tool will process all prompts and determine the best temperature for each.

   3. **Classification Only**:  
      - If you are only interested in classification results, use the model **[Volavion/bert-base-multilingual-uncased-temperature-cls](https://huggingface.co/Volavion/bert-base-multilingual-uncased-temperature-cls)** as per the guidelines.  
      - This model is fine-tuned for temperature classification tasks.

   Once the best temperature is calculated, the **scrolling bar** will automatically be set to this value.  You can manually adjust the temperature if needed to perform inference using your **provided API format**.

   By following these steps, you can effectively use the tool for both single-prompt and batch-prompt operations, as well as for classification tasks.




### Model Selection Guidelines

When selecting a model, if your specific model is **not listed** in the provided options, choose one with a **similar number of parameters**. Research indicates that models with similar parameter counts tend to exhibit **comparable performance variations** when adjusting for temperature changes. To make the best choice, find the model closest to yours in **parameter count**. For instance, if you are using a **Llama 3.1 8B** model, the optimal match would be **Meta-Llama-3-8B-Instruct** from the **Llama 3 series**.

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute the code and its derivatives under the terms of this license. For more details, please refer to the [LICENSE](LICENSE) file.


## Citation

If you use this tool or the associated paper in your work, please cite us using the following format:

To be filled.

