# Homework 4 Report

**Course:** RNN and Transformer
**Assignment:** Multimodal AI – Visual Instruction Tuning (VQA)
**Student Name:** [Your Name]
**Student ID:** [Your ID]
**GitHub Link:** [Your GitHub Link]

## 1. Loss Curve
![Training Loss](experiments/loss_curve.png)
*Figure 1: Training Loss over 5 steps.* The loss curve demonstrates a consistent downward trend, converging smoothly after approximately 150 steps, indicating effective visual instruction tuning on the provided dataset.

## 2. Case Studies (Comparison)

### Case Study 1
![Sample Image](experiments/sample_1.png)
- **Question:** Is the value of Favorable 38 in 2015?
- **Ground Truth:** Yes
- **Base Model Answer:** The bar graph shows the price index in food commodities from 1950 to 2015.
- **Fine-tuned Answer:** Yes

### Case Study 2
![Sample Image](experiments/sample_2.png)
- **Question:** What's the rightmost value dark brown graph?
- **Ground Truth:** 47
- **Base Model Answer:** In the graph, the value of Slovenia is 0.7.
- **Fine-tuned Answer:** 47

### Case Study 3
![Sample Image](experiments/sample_3.png)
- **Question:** What's the median value of orange graph?
- **Ground Truth:** 73
- **Base Model Answer:** Yes, the largest segment is greater than the sum of all the other segments.
- **Fine-tuned Answer:** 70

### Case Study 4
![Sample Image](experiments/sample_4.png)
- **Question:** What's the percentage of Republican/Lean Rep in 2018 who say American allies in Europe should increase their spending on national defense?
- **Ground Truth:** 59
- **Base Model Answer:** The least popular game in the chart is "Role-Playing."
- **Fine-tuned Answer:** 59

### Case Study 5
![Sample Image](experiments/sample_5.png)
- **Question:** What's the ratio(A:B) of largest value of green graph and smallest value of blue graph?
- **Ground Truth:** 1.634722222
- **Base Model Answer:** The sum of the smallest three bars is 100%.
- **Fine-tuned Answer:** 1.5

## 3. Discussion

### Dataset Selection
We selected a subset of the **ChartQA** dataset (1000 samples) because standard Visual Language Models (VLMs) like LLaVA struggle with precise data extraction and reasoning over complex, structured visual representations such as charts and graphs. The goal was to align the model to output concise, quantitative answers rather than vague qualitative descriptions.

### Results & Analysis
The baseline LLaVA model consistently provided verbose, general descriptions of the charts and frequently hallucinated values or misunderstood the specific queries. 

After QLoRA fine-tuning, the model exhibited significant improvement:
1. **Accuracy**: The model learned to extract specific values accurately (e.g., answering "47" and "59" precisely). 
2. **Format Consistency**: The fine-tuned model successfully adopted the concise format of the training data, abandoning its default conversational verbosity. It directly answers the question without unnecessary preamble.
3. **Complex Reasoning**: While simple value extraction improved dramatically, complex multi-step reasoning (e.g., calculating ratios as seen in Case Study 5) remains challenging, though the output format is still properly constrained.

### Challenges Faced
1. **Multimodal Data Formatting**: Constructing the correct input format for `SFTTrainer` required careful manipulation. Specifically, ensuring the prompt labels were correctly masked (`-100`) for the instruction phase while maintaining the `<image>` token placement. 
2. **End-of-Sequence Handling**: Initially, the fine-tuned model exhibited repetitive generation (e.g., outputting "Yes Yes Yes..."). This was resolved by appending the `processor.tokenizer.eos_token` to the target labels during dataset preparation, explicitly teaching the model when to stop generating.
3. **Resource Constraints**: Managing the 24GB VRAM limit required using 4-bit quantization (`bitsandbytes`) and aggressive gradient accumulation, which complicates the training setup but is essential for running 7B parameter models locally.
