# Temperature_eval


This repository serves as a tool to identify the optimal temperature settings for a given prompt and model, leveraging findings from the following paper:

**An In-Depth Study of the Effect of Temperature on Large Language Models. Hot or Cold?**

<p align="center">
  <img src="images/head.jpeg" alt="Empirical Analysis" width="300">
</p>

The tool employs a fine-tuned [fine-tuned BERT model](https://huggingface.co/Volavion/bert-base-multilingual-uncased-temperature-cls) as a prompt classifier to predict the most appropriate capability required to address the input prompt. Based on the classifier's output, the tool then determines the optimal temperature setting, guided by the performance distributions analyzed in the referenced paper. This integration ensures that the temperature configuration aligns with empirical insights, maximizing the effectiveness of the model's response to the prompt.



## Abstract

> When employing Large Language Models (LLMs) on different tasks, a crucial hyperparameter to alternate is the sampling temperature which adjusts the logits of the soft max layers, hence reshaping the distribution \cite{radford2019language}. Recent studies challenge the notion of "Stochastic Parrots" as an analogy for language models, demonstrating that these models can learn formal semantic meaning rather than merely memorizing data. Randomness plays a critical role in LLMs, driving research on temperature in so-called "Next Token Generation". The researcher studied how LLM performance changes with temperature on complex datasets requiring multiple abilities. However, there remains a lack of comprehensive analysis that independently evaluates individual model skill over a broad range of temperature settings. In this paper, we systematically examine the impact of temperature on datasets specifically designed to evaluate six distinct skills. Our study spans Small (0B-3B), Medium (6B-13B), and Large (50B-80B) open-source models, assessing their performance across a temperature range from 0 to 2. The results reveal nuanced and skill-specific variations in model responses to temperature changes. To address the practical demands of complex real-world applications of LLMs, we introduce a BERT-based optimal temperature selector that dynamically adjusts temperature, giving a prompt. This approach achieves a performance improvement exceeding 5\% for mediums and small size models compared to fixed-temperature configurations. Furthermore, we investigate the robustness of temperature effects under conditions of full-precision models and also extend the analysis on three models to temperatures up to 4.0. Our findings confirm consistent temperature effect with performance degradation, and the "Mutation Temperature"—the point where significant performance shifts occur—tends to increase with model size.

## Abilities
This respository mainly contains several LLM abilities evaluations include: 

- **Causal Reasoning (CR)**: A cognitive faculty historically ascribed solely to humans that derive conclusions from given premises by adhering to strict logical principles.
- **Creativity (CT)**: An ability defined involves generating novel and valuable ideas, concepts, or products which require both originality and effectiveness.
- **Instruction Following (IF)**: This reflects the crucial ability to adhere to instructions presented in prompts, particularly important in the application of LLMs.
- **In-Context Learning (ICL)**: The emerging verified ability reflects the skill to comprehend text and perform tasks within its context and few examples, and this skill has become a new paradigm for natural language processing (NLP).
- **Summarization (SUM)**: This entails condensing lengthy texts or discussions into concise and informative summaries, while preserving key information and main ideas.
- **Machine Translation (MT)**: MT is a subfield of computational linguistics, and LLMs have shown outstanding potential in translating text from one language to another.

---


## Setting Up the Environment

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

3. **Start the Tool**  
   Launch the server by running the main script:  
   ```bash
   python main_start.py
   ```

4. **Access the Tool**  
   After the server starts, a link will appear in the terminal. Click on the link to access and interact with the tool.


---

## Tool Usage Overview

This section provides a structured guide for understanding and interacting with the tool.

### 1. **Prompt Input Section**  
   Users can input a prompt in this section and select the desired model for evaluation.

### 2. **Action**  
   After entering the prompt and selecting the model, click the **"Analyze"** button to initiate the evaluation process.

### 3. **Model Invocation**  
   The tool utilizes the model **Volavion/bert-base-multilingual-uncased-temperature-cls** available on Hugging Face. This model is pre-trained to analyze input prompts and fine-tuned for temperature classification tasks. No special permissions are required for downloading the model.

### 4. **Class Label Output**  
   Upon clicking the **"Analyze"** button:
   - A bar chart visualizing the probability distribution across various class labels will be displayed.
   - These probabilities are generated using the fine-tuned BERT-based model.

### 5. **Best Temperature Display**  
   The **"Best Temperature"** text box will:
   - Display the optimal temperature based on the predicted class label.
   - This temperature is determined using the performance distribution referenced in the associated research paper.

### 6. **Input Temperature Adjustment**  
   - The recommendation section includes a sliding bar, which is automatically set to the best temperature identified during analysis.
   - Users can manually adjust the temperature to accommodate specific requirements.

### 7. **API Key Setup**  
   - Users need to provide an API key for generation tasks.
   - The tool supports API keys in the **vLLM OpenAI API** format as well as **Ollama API keys**.

### 8. **Outputs Section**  
   - If a single prompt is provided as input, the result will be displayed directly.
   - For CSV input files:
     - The file should include at least one column named **"input"**, which contains the prompts for evaluation.
     - The output CSV file will include an additional column named **"generated response"** that contains results generated using the specified API.
     - A download button is provided for easy retrieval of the processed file.

---

### Model Selection Guidelines

- **Model Selection**:  
   If the model you are using is **not listed** in the provided options, select a model with a **similar number of parameters**.
   
- **Reasoning**:  
   Research shows that models with a similar number of parameters exhibit **similar performance variations** with respect to temperature changes.

- **How to Choose**:
   - Select the model that most closely matches yours in terms of **parameter count**.
   - For example, if using a **Llama 3.1 8B** model, choose the closest match within the **Llama 3 series**.
     - In this case, the optimal choice would be: **Meta-Llama-3-8B-Instruct**

---


---

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute the code and its derivatives under the terms of this license. For more details, please refer to the [LICENSE](LICENSE) file.

---

## Citation

If you use this tool or the associated paper in your work, please cite us using the following format:

To be filled.

