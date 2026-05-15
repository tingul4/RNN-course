# Assignment 4: Multimodal AI – Visual Instruction Tuning (VQA)

**Student ID:** 314832008<br>
**Source Code:** [https://github.com/tingul4/RNN-course](https://github.com/tingul4/RNN-course)

## Part 1: Baseline Inference

### 1. Model Setup
We loaded the pre-trained `llava-hf/llava-1.5-7b-hf` model. To fit the 7B parameter model within the 24GB VRAM constraint of the provided environment, we applied 4-bit quantization using `BitsAndBytesConfig`. Specifically, we enabled `load_in_4bit=True`, set `bnb_4bit_quant_type="nf4"`, enabled double quantization (`bnb_4bit_use_double_quant=True`), and set the compute dtype to `torch.bfloat16` to maintain numerical stability during inference and training. The model was loaded with `device_map="auto"` and `attn_implementation="eager"`.

### 2. Zero-Shot Testing
We selected 5 images from the **ChartQA** dataset, which contains complex charts and graphs. When asking the baseline LLaVA model specific questions about values in these charts, we observed that it frequently gave vague, general descriptions of the image rather than extracting the precise numerical answers requested. It often failed to follow the implied format constraint of ChartQA (which expects concise quantitative answers). These "Before" responses are documented alongside the "After" responses in Part 3.

## Part 2: Visual Instruction Tuning with QLoRA

### 1. Dataset Preparation
- **Dataset**: We utilized a subset of 1000 samples from the `HuggingFaceM4/ChartQA` dataset.
- **Formatting**: The image-text pairs were formatted into the standard VLM conversation format. We mapped the `query` to the user prompt (along with the `<image>` token) and the `label` to the assistant's response. Crucially, we appended the `EOS` token to the assistant's completion to teach the model to stop generating text after outputting the answer.

### 2. Fine-tuning (The Training Loop)
Before applying adapters, we prepared the quantized model using `prepare_model_for_kbit_training(model)` to cast necessary layers to float32 for stability. We then used the **PEFT (Parameter-Efficient Fine-Tuning)** library from Hugging Face to apply LoRA adapters to the Language Model while keeping the Vision Encoder (ViT) frozen. 

Our `LoraConfig` was defined using PEFT with the following hyperparameters:
- **Rank (`r = 16`)**: Controls the dimensionality of the low-rank matrices. A rank of 16 provides a solid balance, keeping the number of trainable parameters extremely low (~1-2%) while still offering enough capacity for the model to effectively learn the ChartQA domain.
- **Alpha (`lora_alpha = 32`)**: The scaling factor for the LoRA update. Setting alpha to twice the rank (`r * 2`) is a widely adopted heuristic that stabilizes gradients and ensures the initial LoRA updates are sufficiently weighted without overwhelming the base model's knowledge.
- **Dropout (`lora_dropout = 0.05`)**: A small 5% dropout probability is applied to the LoRA matrices to mitigate overfitting, which is especially important given our relatively small dataset of only 1000 samples.
- **Target Modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`)**: These denote the Query, Key, Value, and Output projection matrices within the LLM's self-attention mechanism. Targeting all four attention projection layers (rather than just `q_proj` and `v_proj`) gives the model maximum flexibility to adapt its attention patterns and reasoning pathways for the complex task of chart comprehension.
- **Task Type**: `CAUSAL_LM`

We utilized the `SFTTrainer` from the `trl` library, which natively handles multimodal data collation and sets prompt labels to `-100` for completion-only loss computation. The model was trained with a learning rate of 2e-4 and a batch size of 4 (with gradient accumulation steps of 4).

![Training Loss](experiments/loss_curve.png)
*Figure 1: Training Loss Curve.* The loss curve demonstrates a consistent downward trend, indicating effective visual instruction tuning on the ChartQA domain.

## Part 3: Evaluation & Analysis

### 1. Inference with Adapter
After training, we loaded the base LLaVA model along with our trained LoRA adapter to evaluate its performance on the same 5 images from Part 1.

<div style="page-break-after: always;"></div>

### 2. Comparison & Visualization (Case Studies)

#### Case Study 1
![Sample Image](experiments/sample_1.png)
- **Question:** Is the value of Favorable 38 in 2015?
- **Ground Truth:** Yes
- **Base Model Answer (Before):** The bar graph shows the price index in food commodities from 1950 to 2015.
- **Fine-tuned Answer (After):** Yes

<div style="page-break-after: always;"></div>

#### Case Study 2
![Sample Image](experiments/sample_2.png)
- **Question:** What's the rightmost value dark brown graph?
- **Ground Truth:** 47
- **Base Model Answer (Before):** In the graph, the value of Slovenia is 0.7.
- **Fine-tuned Answer (After):** 47

<div style="page-break-after: always;"></div>

#### Case Study 3
![Sample Image](experiments/sample_3.png)
- **Question:** What's the median value of orange graph?
- **Ground Truth:** 73
- **Base Model Answer (Before):** Yes, the largest segment is greater than the sum of all the other segments.
- **Fine-tuned Answer (After):** 70

<div style="page-break-after: always;"></div>

#### Case Study 4
![Sample Image](experiments/sample_4.png)
- **Question:** What's the percentage of Republican/Lean Rep in 2018 who say American allies in Europe should increase their spending on national defense?
- **Ground Truth:** 59
- **Base Model Answer (Before):** The least popular game in the chart is "Role-Playing."
- **Fine-tuned Answer (After):** 59

<div style="page-break-after: always;"></div>

#### Case Study 5
![Sample Image](experiments/sample_5.png)
- **Question:** What's the ratio(A:B) of largest value of green graph and smallest value of blue graph?
- **Ground Truth:** 1.634722222
- **Base Model Answer (Before):** The sum of the smallest three bars is 100%.
- **Fine-tuned Answer (After):** 1.5

### Analysis
- **Did the answer improve?** Yes. The baseline model failed to extract correct values, offering generic chart descriptions instead. The fine-tuned model accurately extracts specific values and correctly answers yes/no questions based on the visual data.
- **Is the format more consistent?** Yes. The fine-tuned model successfully learned to output concise, precise answers matching the ground truth format, abandoning its default conversational verbosity.

<div style="page-break-after: always;"></div>

## Discussion

### Why did you choose this specific dataset?
We selected the **ChartQA** dataset because it represents a clear domain gap for standard VLMs. While general VLMs excel at natural image description, they struggle with precise data extraction and reasoning over structured visual representations like charts. Fine-tuning on ChartQA clearly demonstrates the power of QLoRA to adapt a general model to a highly specific, quantitative domain.

### What challenges did you face with multimodal data loading?
1. **Multimodal Collation**: Formatting the multimodal input correctly for the `SFTTrainer` was challenging. We had to ensure the dataset was mapped to a specific dictionary structure (`prompt`, `completion`, `images`) so that the trainer could automatically handle tokenization, chat templates, and masking (`-100` for prompts).
2. **End-of-Sequence Handling**: Initially, the fine-tuned model suffered from repetitive generation (e.g., repeating "Yes Yes Yes..."). We resolved this by explicitly appending the `processor.tokenizer.eos_token` to the target labels during dataset formatting, ensuring the model learned to terminate generation after outputting the concise answer.
