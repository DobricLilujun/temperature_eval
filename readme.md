# Temperature_eval

There will be some updates for the in-depth analysis on the single inference.


https://medium.com/ai-assimilating-intelligence/cross-entropy-in-large-language-models-llms-4f1c842b5fca
   
**Detecting grammer error checking? **
**Detecting hallucination error checking? **


This is a respository for evaluation and visulization of the results obtaining from observations. This is the respository acoomplanied by the following paper:

> **Hot N Cold: A Comprehensive Analysis of Temperature on the Performance of Llms**

<p align="center">
  <img src="images/head.jpeg" alt="empirical" width="300">
</p>


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



## How to use it?

### Install Dependencies
```bash
conda create -n env_temperature_eval python=3.10

pip install gradio transformers torch matplotlib
```

### Tool Usage

Here’s a clearer, more structured version of your explanation using Markdown:

---

### Main Interface Overview

1. **Prompt Input Section**:  
   Users can enter a prompt here and select the model they are using.

2. **Action**:  
   After entering the prompt and selecting the model, click the "Analyze" button.

3. **Model Invocation**:  
   The system will automatically invoke the **Volavion/bert-base-multilingual-uncased-temperature-cls** from huggingface that has been trained to analyze the input prompt.

3. **Class Label Output**:  
  The system will automatically invoke the **BERT model** that has been trained to analyze the input prompt.


---

### Model Selection Guidelines

- **Model Selection**:  
   If the model you are using is **not listed** in the provided options, select a model with a **similar number of parameters**.
   
- **Reasoning**:  
   Research shows that models with a similar number of parameters exhibit **similar performance variations** with respect to temperature changes.

- **How to Choose**:
   - Select the model that most closely matches yours in terms of **parameter count**.
   - For example, if using a **Llama 3.1 8B** model, choose the closest match within the **Llama 3 series**.
     - In this case, the optimal choice would be:
       - **Meta-Llama-3-8B-Instruct**
       - **Llama-3.1-8B-Instruct**

---
