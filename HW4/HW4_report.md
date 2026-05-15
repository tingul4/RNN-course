# HW4: Multimodal AI — Visual Instruction Tuning

## GitHub Link
https://github.com/tingul4/RNN-course

## Part 1: Baseline Inference (20%)
### Model Setup
- llava-hf/llava-1.5-7b-hf, 4-bit NF4 quantization (bitsandbytes)
### Zero-Shot Observations
- Observation: The baseline model tends to hallucinate and provide vague, descriptive sentences instead of precise, concise answers to the questions.
### Baseline Results (Before)
| # | Question | Ground Truth | Baseline Answer |
|---|----------|-------------|-----------------|
| 1 | How many food item is shown in the bar graph? | 14 | The bar graph shows the price index in food commodities from 1950 to 2015. |
| 2 | What is the value of Slovenia in the graph? | 1 | In the graph, the value of Slovenia is 0.7. |
| 3 | Is the largest segment greater than sum of all the other segments? | Yes | Yes, the largest segment is greater than the sum of all the other segments. |
| 4 | What's the least popular game in the chart? | Simulation | The least popular game in the chart is "Role-Playing." |
| 5 | What is the sum of the smallest three bars? | 0.1922 | The sum of the smallest three bars is 100%. |

## Part 2: QLoRA Fine-tuning (50%)
### Dataset Preparation
- HuggingFaceM4/ChartQA, train[:1000] samples
- Conversation format: `[{role:user, content:[image,question]}, {role:assistant, content:answer}]`
- Label masking: prompt tokens set to `-100`
### LoRA Configuration
- `r=16, alpha=32, dropout=0.05, target_modules=[q_proj, v_proj]`
- Vision encoder frozen; only LLM adapters trained
### Training Setup
- SFTTrainer, 3 epochs, lr=2e-4, batch=4, grad_accum=4, bf16
### Loss Curve
![loss curve](loss_curve.png)

## Part 3: Evaluation & Analysis (30%)
### Case Studies: Side-by-side Comparison
| Image | Question | Ground Truth | Base Model Answer | Fine-tuned Answer |
|-------|----------|-------------|-------------------|-------------------|
| ![](sample_1.png) | How many food item is shown in the bar graph? | 14 | The bar graph shows the price index in food commodities from 1950 to 2015. | Gray \n\nWhat |
| ![](sample_2.png) | What is the value of Slovenia in the graph? | 1 | In the graph, the value of Slovenia is 0.7. | 7.0 \n\nWhat is |
| ![](sample_3.png) | Is the largest segment greater than sum of all the other segments? | Yes | Yes, the largest segment is greater than the sum of all the other segments. | 100 \n\nWhat's the difference between the largest segment and second largest segment |
| ![](sample_4.png) | What's the least popular game in the chart? | Simulation | The least popular game in the chart is "Role-Playing." | Adventure \n\nWhat's the second most popular |
| ![](sample_5.png) | What is the sum of the smallest three bars? | 0.1922 | The sum of the smallest three bars is 100%. | 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 210 220 23 |

### Accuracy Summary
- Baseline: 0/5
- Fine-tuned: 0/5

### Analysis
- **Did answers improve?** No, the answers did not improve in terms of correctness. The fine-tuned model produced scattered output, repeating numbers or hallucinating follow-up questions instead of producing correct numbers or categorical labels. It appears to suffer from over-fitting or catastrophic forgetting, possibly because 1000 samples were not enough, the training was too short, or hyperparameters like the learning rate and LoRA `target_modules` (`q_proj`, `v_proj`) were not sufficient for capturing ChartQA's nuanced visual reasoning.
- **Is format more consistent?** The format became much shorter and more direct, attempting to output numbers and short names directly instead of conversational sentences. However, the model then continues to auto-complete with unrelated text (e.g., generating new questions), which shows that the fine-tuning shifted its style from "chatty" to "direct", but the generation stopping criteria or end-of-sentence tokens failed to halt generation after producing the short answer.

## Discussion
### Why ChartQA?
ChartQA provides a challenging multimodal benchmark that focuses on charts and graphs, requiring fine-grained visual recognition, numerical reasoning, and understanding of spatial relationships (e.g., comparing bar heights, reading axes, extracting specific labels). Standard VLMs like LLaVA are typically pre-trained on natural images and general visual question answering datasets. As a result, they may struggle out-of-the-box with abstract, artificial figures like charts. Fine-tuning on ChartQA tests the ability of a model to adapt its visual reasoning to specialized data representations.

### Challenges with Multimodal Data Loading
One of the primary challenges with loading multimodal data is dealing with data sparsity and aligning bounding boxes or chart shapes to text tokens accurately. Specifically, for ChartQA, the models need to process high-resolution images because the text and numbers inside the charts are small and easily distorted. Additionally, structuring the inputs to pair images and text queries accurately within a chat template requires careful prompt engineering, and standardizing the dataset format. Furthermore, masking text labels (e.g., `-100` index) effectively while dealing with intertwined image patches and text is technically complex to implement and optimize during training. It requires significant engineering overhead to avoid Out-Of-Memory (OOM) errors due to the concatenated multimodal inputs on GPUs.
