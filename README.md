---
language:
- hi
- en
- as
- bn
- brx
- doi
- gu
- kn
- ks
- mai
- ml
- mni
- mr
- ne
- or
- pa
- sa
- sat
- sd
- ta
- te
- ur
pipeline_tag: text-generation
library_name: transformers
tags:
- mixture-of-experts
- multilingual
- indian-languages
---

<div align="center">
  <img src="./BharatGen Logo.png" width="60%" alt="BharatGen" />
</div>
<hr>
<div align="center">
  <a href="https://huggingface.co/bharatgenai/Param-2-17B-MoE-A2.4B/blob/main/LICENSE" target="_blank" style="margin: 4px;">
    <img alt="License" src="https://img.shields.io/badge/License-BharatGen%20Non--Commercial-yellow.svg" />
  </a>
</div>

# Param-2-17B

**BharatGen** presents **Param-2-17B-MoE-A2.4B**, a large-scale Mixture-of-Experts (MoE) language model designed to deliver high model capacity while retaining the inference efficiency of a much smaller dense model. It uses a **Hybrid MoE architecture** with **17B total parameters**, while activating only **2.4B parameters per token**.

The model is pretrained from scratch, with a strong emphasis on **linguistic diversity**, **cultural grounding**, and **multilingual representation**, particularly for Indian languages. It is released as an **early post-training checkpoint** with advanced capabilities including reasoning, tool calling, mathematics, and code generation, making it suitable for diverse downstream applications and further fine-tuning.

---

## 🌟 Key Highlights

- **17B parameter** Mixture of Experts (MoE) language model
- **Multilingual**: English, Hindi + 21 Indian languages
- **Trained on ~22 trillion tokens** across two pretraining phases
- **Uses 64 specialized experts**, dynamically activated per token
- **Supports long-context understanding** (up to 4096 tokens)
- **Efficient inference**: Only 2.4B active parameters per token
- Advanced Capabilities: **Thinking & Reasoning, Tool Calling, Mathematics, Code Generation**
- Designed for diverse downstream applications and further fine-tuning

---

## 🗣️ Supported Languages

In addition to **English** and **Hindi**, the model supports the following **21 Indian languages**:

| 🌏 Languages |
|--------------|
| Assamese • Bengali • Bodo <br> Dogri • Gujarati • Kannada <br> Konkani • Kashmiri • Maithili <br> Malayalam • Manipuri • Marathi <br> Nepali • Oriya • Punjabi <br> Sanskrit • Santali • Sindhi <br> Tamil • Telugu • Urdu |

This broad language coverage enables stronger performance on **region-specific tasks** and promotes **inclusive representation** of India’s diverse and underrepresented linguistic communities.

---

## 🏗️ Model Architecture

The model utilizes a **Hybrid MoE approach**. Unlike traditional MoE models, this architecture uses a combination of **Shared Experts** (always active) and **Routed Experts** (activated selectively) to ensure both general knowledge stability and specialized task performance.

| Metric | Value |
|--------|-------|
| **Architecture** | Hybrid MoE |
| **Total Parameters** | 17B |
| **Active Parameters/Token** | 2.4B |
| **Total Experts** | 64 |
| **Top-K Experts** | 6 |
| **Shared Experts** | 2 |
| **Hidden Size** | 2048 |
| **Intermediate Size** | 7168 |
| **Max Context Length** | 4096 tokens |
| **Vocabulary Size** | 128,000 |
| **Activation Function** | SiLU |
| **Normalization** | RMSNorm |
| **Num. Hidden Layers** | 21 |
| **Attention Heads** | 32 (Query) / 8 (KV) |
| **Attention Mechanism** | Grouped Query Attention |
| **Precision** | bf16-mixed |

### Architecture Innovation

Our Hybrid MoE architecture introduces two key innovations: (1) a dense layer before sparse MoE layers that transforms raw inputs into structured representations for improved stability, and (2) Two shared routing experts that remain always active alongside 6 dynamically routed experts. The shared experts specifically address Indian language diversity by maintaining stable cross-lingual representations, enabling smooth code-switching, and facilitating knowledge transfer across all 23 supported languages.

---

## ✨ Key Features

- **Multilingual-First Design**: Pretrained to support English, Hindi, and 21 Indian languages with better regional linguistic coverage.
- **Shared + Routed Experts**: Separates general language understanding from language-specific specialization for improved multilingual performance.
- **Efficient Sparse Scaling**: 17B total parameters with 2.4B active parameters per token, enabling high capacity with efficient inference.
- **Cross-Lingual Representation Learning**: Improves performance on low-resource Indian languages through shared multilingual knowledge.
- **Large-Scale Multilingual Pretraining**: Trained on ~22 trillion tokens emphasizing linguistic diversity and culturally grounded data.
- **Long-Context Multilingual Processing**: Supports up to 4096 tokens for improved understanding of long-form and multi-turn content. (A longer-context variant of this model, supporting up to 128k context length, will be released soon.)

---

## 📚 Training Data

### Dataset Composition

#### **PT-1: Pre-Training Phase 1**

| Data Type | Tokens |
|-----------|--------|
| English | 8.5T |
| Hindi | 998.6B |
| Math | 374.8B |
| Code | 1.83T |
| Multilingual Indian Languages | 1.5T |
| Technical | 500B |
| Instruction | 351B |
| **Total Phase 1** | **14.05T** |

#### **PT-2: Pre-Training Phase 2**

| Data Type | Tokens |
|-----------|--------|
| English | 2.5T |
| Math | 773B |
| Code | 2.46T |
| Multilingual Indian Languages | 1T |
| Instruction | 740B |
| Reasoning | 607B |
| **Total Phase 2** | **8.08T** |

**Combined Total Training Data**: **22.13 Trillion Tokens**

---

## 🎯 Pre-Training Details

- **Total Training Tokens**: ~22 Trillion 
- **Data Quality**: Highly curated using 22 Indian language quality classifiers along with an in-house preprocessing pipeline.
- **Scheduler**: Cosine Annealing
- **Training Framework**: NVIDIA NeMo
- **Hardware**: NVIDIA H100 GPUs
- **Total Training Duration**: 24 Days
- **Precision**: bf16-mixed

---

## 🌍 Environmental Impact

### Carbon Emissions

Training large-scale foundation models requires substantial computational resources. In the interest of transparency and responsible AI development, we provide an estimated carbon footprint of our training process.
Training Configuration

| Metric | Value | 
|--------|-------|
| **Total GPU Hours** | 576 Hours |
| **Hardware Used** | NVIDIA H100 GPUs |
| **Estimated Power Usage** | ~258,048 kWh (approximate) | 
| **Carbon Footprint** | ~194 metric tons CO₂eq |

> **Note**:
* Energy estimation is based on GPU TDP with a 25% infrastructure overhead.
* Carbon intensity reflects the India national grid average (0.75 kg CO₂eq/kWh).
* Actual emissions may vary depending on data center efficiency (PUE), renewable energy mix, and workload utilization.
* Calculations follow standard ML CO₂ estimation practices and are aligned with methodologies used in ML CO₂ Impact reporting.

### Efficiency Measures

- **Sparse Activation**: Only 2.4B out of 17B parameters are activated per token, reducing computational overhead during inference
- **Efficient Architecture**: Hybrid MoE design optimizes compute-to-performance ratio
- **Multi-Phase Training**: Structured training phases ensure efficient use of computational resources

---

## 🚀 Model Inference

> **Requirements:**
> - Transformers version - 4.52.3
> - Support for Transformers v5 coming soon

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from parsers import parse_model_output

model_name = "bharatgenai/Param2-17B-A2.4B-Thinking"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=False
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="auto"
)

conversation = [
    {"role": "system", "content": "You are helpful assistant."},
    {"role": "user", "content": "What is the BharatGen Mission?"}
]

inputs = tokenizer.apply_chat_template(
    conversation=conversation,
    return_tensors="pt",
    add_generation_prompt=True
).to(model.device)

with torch.no_grad():
    output = model.generate(
        inputs,
        max_new_tokens=300,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.7,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=False,
    )

generated_tokens = output[0][inputs.shape[-1]:]

# 🔥 IMPORTANT: skip_special_tokens=False
generated_text = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=False
)

parsed = parse_model_output(generated_text)

print("\n========== RAW ==========\n", generated_text)
print("\n========== REASONING ==========\n", parsed["reasoning"])
print("\n========== TOOL CALLS ==========\n", parsed["tool_calls"])
print("\n========== FINAL ANSWER ==========\n", parsed["final_answer"])
```
Note: The parser is available within the model directory. You may download and customize it according to your use case.

---

## ✅ Quick Guide for Best Results

# Use Deterministic Decoding for Consistency
Set do_sample = False to get stable, repeatable, and more reliable outputs. This is recommended for evaluations, production use, and structured tasks.

# Prompt Tuning is Essential
Clearly specify the task, format, and constraints in your prompt. Well-structured instructions significantly improve accuracy and relevance for specific use cases.

# Leverage Strong Structured Output Capabilities
The model performs well at generating Markdown, tables, JSON, and other structured formats, making it ideal for documentation, reports, and formatted outputs.

---

# 🌟 Model Capabilities & Examples

### 🧮 Example: Mathematical Reasoning (Chain-of-Thought)

#### **Prompt**
> **A class has 30 students.**
>
> **The average score of the whole class in a test is 70.**
> **The average score of the top 10 students is 85.**
> **The average score of the bottom 10 students is 50.**
>
> **👉 What is the average score of the middle 10 students?**

#### **Model Response**

<details>
<summary><strong>🧠 Click to view Reasoning Process</strong></summary>

&lt;think&gt;
# Ahh.. Thinking about the problem..

Okay, let's see. The problem is about a class of 30 students. The average score of the whole class is 70. The top 10 students have an average of 85, and the bottom 10 have an average of 50. We need to find the average score of the middle 10 students.

First, I need to recall that the average score is the total sum of all scores divided by the number of students. So, for the entire class, the total sum of scores would be 30 students multiplied by 70, which is 2100. That makes sense because average is sum divided by number.

Now, the top 10 students have an average of 85. So their total sum would be 10 times 85, which is 850. Similarly, the bottom 10 students have an average of 50, so their total sum is 10 times 50, which is 500. 

But here's the thing: the class is divided into three groups—top, middle, and bottom. Each group has 10 students. So, the total sum of all three groups should equal the total sum of the entire class. That is, 850 (top) + sum of middle 10 + 500 (bottom) = 2100.

To find the sum of the middle 10 students, we subtract the top and bottom sums from the total. Let's compute that: 2100 minus 850 minus 500. That would be 2100 - 850 = 1250, then 1250 - 500 = 750. So the middle 10 students have a total sum of 750.

Now, to find their average, we take that sum and divide it by the number of students in the middle group, which is 10. So 750 divided by 10 equals 75. Therefore, the average score of the middle 10 students is 75.

Wait, let me check my math again to make sure. 

Total class sum: 30 * 70 = 2100. Correct.

Top 10 sum: 10 * 85 = 850. Bottom 10 sum: 10 * 50 = 500. Adding those: 850 + 500 = 1350. 

Subtracting that from the total: 2100 - 1350 = 750. Divided by 10 students: 750 / 10 = 75. Yep, that seems right. 

I don't think there's any overlap here because the top and bottom groups are separate, and the middle group is the remaining 10. So this should be straightforward. So the answer is 75. 

Hmm, but just to think again, sometimes in these problems, there might be some trick where the middle group is not exactly 10 students, but in this case, the problem states the class has 30 students, divided into top 10, middle 10, bottom 10. So they add up to 30. So each is 10. Therefore, the logic holds. So the average of the middle 10 is 75. So I think that's the answer.
&lt;/think&gt;

</details>

The class has 30 students with an average score of 70. The total sum of all scores is \(30 \times 70 = 2100\). 

The top 10 students have an average score of 85, giving a total of \(10 \times 85 = 850\). The bottom 10 students have an average score of 50, giving a total of \(10 \times 50 = 500\). 

Subtracting the top and bottom sums from the total gives the middle 10 students' total: \(2100 - 850 - 500 = 750\). 

Dividing this by 10 students gives the average score of the middle 10 students: \(\frac{750}{10} = 75\).

The average score of the middle 10 students is \(\boxed{75}\).

---

### 🇮🇳 Example: Multilingual Coding (Telugu)

#### **Prompt**
> **Write a binary search program in Python. All comments in the code must be written in Telugu language.**

#### **Model Response**

<details>
<summary><strong>🐍 Click to view Reasoning Process</strong></summary>

&lt;think&gt;
## Reasoning
User binary search కోసం Python program కోరారు. Comments అన్ని Telugu లో ఉండాలి. Binary search సాధారణ algorithm, array, low, high, mid. Telugu లో వివరణలు ఇవ్వాలి. Code లో comments Telugu లో ఉండాలి, code logic correct ఉండాలి. Python లో function name in English, comments Telugu.
&lt;/think&gt;

</details>

```python
def binary_search(arr, target):
    # ఈ ఫంక్షన్ లక్ష్య విలువను కనుగుకోవడానికి binary search ను అమలు చేస్తుంది
    low = 0
    high = len(arr) - 1
    
    while low <= high:
        mid = (low + high) // 2
        
        # మధ్య విలువను పరిశీలిస్తాము
        if arr[mid] == target:
            # లక్ష్య విలువను కనుగొన్నాం, దాని index ను తిరిగి ఇస్తాం
            return mid
        elif arr[mid] < target:
            # లక్ష్య విలువ పెద్దది, కాబట్టి low ను పెంచుతాం
            low = mid + 1
        else:
            # target చిన్నది, కాబట్టి high ను తగ్గిస్తాం
            high = mid - 1
            
    # లక్ష్య విలువ లేదు, -1 ను తిరిగి ఇస్తాం
    return -1

# ఉదాహరణ వినియోగం
arr = [2, 3, 5, 7, 11, 13]
target = 7
result = binary_search(arr, target)
print(f"లక్ష్య విలువ {target} యొక్క index: {result}")
```

### 🌏 Example: Multilingual Cultural Question (Hindi)

#### **Prompt**
> **भारत अपनी विविध संस्कृति के लिए जाना जाता है। भारत में अलग-अलग राज्यों की भाषाएँ, भोजन और त्योहार क्यों अलग होते हैं? इस सांस्कृतिक विविधता का भारतीय समाज पर क्या सकारात्मक प्रभाव पड़ता है?**

#### **Model Response**

<details>
<summary><strong>💭 Click to view Reasoning Process</strong></summary>

&lt;think&gt;
## प्रश्न की समझ
उपयोगकर्ता ने भारत की विविध संस्कृति के कारण राज्य‑विशिष्ट भाषाओं, भोजन और त्योहारों में अंतर के बारे में पूछा है तथा इस विविधता के सामाजिक प्रभाव पर चर्चा चाहता है। इसलिए उत्तर में दो भाग चाहिए: (1) सांस्कृतिक विविधता के कारणों का स्पष्टीकरण, (2) इस विविधता के सकारात्मक प्रभावों की सूची।

## मुख्य कारणों की पहचान
1. **भौगोलिक विविधता** – पहाड़, मैदान, समुद्र तट, रेगिस्तान आदि विभिन्न जलवायु और स्थल बनाते हैं, जिससे अलग‑अलग कृषि‑उत्पादन और खाद्य‑प्रणाली विकसित होती है।
2. **जनजातीय एवं भाषा विविधता** – विभिन्न जनजातीय समूह, उनके मूल भाषाएँ, लिपियाँ और सांस्कृतिक परम्पराएँ राज्य‑स्तर पर भाषा का प्रयोग निर्धारित करती हैं।
3. **ऐतिहासिक एवं राजनैतिक प्रभाव** – विभिन्न राजवंश, उपनिवेश, और स्वतंत्रता के बाद के राज्य‑निर्माण ने प्रशासनिक सीमाएँ और सांस्कृतिक पहचान को अलग किया।
4. **धर्म और उत्सव परंपरा** – प्रत्येक क्षेत्र में प्रमुख धर्म (हिंदू, मुस्लिम, सिख, बौद्ध आदि) और उनके स्थानीय उत्सव, रीति‑रिवाज अलग‑अलग विकसित होते हैं।
5. **स्थानीय संसाधन एवं परम्पराएँ** – प्राकृतिक संसाधन (जैसे जल, खनिज, समुद्र) और स्थानीय कारीगर, हस्तशिल्प आदि सांस्कृतिक अभिव्यक्ति को आकार देते हैं।

## उत्तर का ढांचा तय करना
- पहले कारणों को क्रमबद्ध बिंदुओं में प्रस्तुत करें, जिससे उपयोगकर्ता को स्पष्ट समझ मिले।
- फिर प्रत्येक कारण के बाद तत्काल परिणाम (भाषाएँ, भोजन, त्योहार) को संक्षिप्त रूप में जोड़ें, जैसा कि संदर्भ उत्तर में किया गया है।
- अंत में सामाजिक प्रभाव को पाँच प्रमुख बिंदुओं में विभाजित करें, प्रत्येक बिंदु को अलग‑अलग उप‑बिंदु के साथ विस्तृत करें, और अंत में एक समग्र निष्कर्ष दें जो विविधता को एकता के साथ जोड़े।

## सकारात्मक प्रभावों की सूची बनाना
1. **सांस्कृतिक अभिव्यक्ति** – विविधता के कारण विभिन्न कला, संगीत, नृत्य, साहित्य आदि का विकास।
2. **आर्थिक प्रभाव** – पर्यटन, हस्तशिल्प, स्थानीय बाजारों की विविधता, जिससे आय उत्पन्न होती है।
3. **शिक्षा एवं अनुसंधान** – विभिन्न भाषा‑साहित्य, इतिहास, सामाजिक विज्ञान के अध्ययन के अवसर।
4. **वैश्विक मंच पर पहचान** – अंतरराष्ट्रीय मंच पर विविध सांस्कृतिक उत्पादों का प्रदर्शन, जिससे भारत की छवि मजबूत होती है।
5. **सामाजिक मूल्य एवं सहिष्णुता** – विविधता के माध्यम से विभिन्न धर्म, जातीय समूहों के बीच सहिष्णुता और सहयोग का विकास।

## उत्तर का स्वर और शैली
संदर्भ उत्तर में सरल, बिंदु‑आधारित शैली है, प्रत्येक बिंदु को बोल्ड शीर्षक के साथ प्रस्तुत किया गया है। इसलिए इसी शैली को अपनाते हुए, कारणों को क्रमांकित सूची में लिखें और प्रभावों को भी समान रूप में प्रस्तुत करें। यह उपयोगकर्ता की अनुरोध के अनुसार उत्तर को स्पष्ट, संगठित और सकारात्मक बनाता है।

## अंतिम रूपरेखा
- परिचयात्मक वाक्य: “भारत अपनी विविध संस्कृति के लिए जाना जाता है, इसलिए प्रत्येक राज्य में अलग‑अलग भाषाएँ, भोजन और त्योहार होते हैं।”
- कारणों की सूची (1‑5) और उनके तुरंत प्रभाव।
- सकारात्मक प्रभावों की सूची (1‑5) के साथ विस्तृत बिंदु।
- निष्कर्ष: विविधता एकता के साथ समृद्धि लाती है।

यह सोच प्रक्रिया सीधे संदर्भ उत्तर की सामग्री और संरचना की ओर ले जाती है, जिससे वही उत्तर उत्पन्न होगा।
&lt;/think&gt;

</details>

भारत अपनी विविध संस्कृति के लिए जाना जाता है, इसलिए प्रत्येक राज्य में अलग- अलग भाषाएँ, भोजन और त्योहार होते हैं।

1. **भिन्न भौगोलिक स्थिति और जलवायु** भारत में पहाड़, मैदान, जंगल, समुद्र तट और रेगिस्तान जैसे विविध भू-भाग हैं। इसलिए प्रत्येक राज्य में अलग-अलग जलवायु है और अलग-अलग कृषि उत्पादन होता है। यह अलग-अलग भोजन और खाने की परंपराओं को जन्म देता है। 

2. **भाषाओं और जातीय विविधता** भारत में विविध जातीय समूह और भाषाएँ हैं। प्रत्येक राज्य में अपनी भाषा है, जो उस राज्य की पहचान है। भाषा के कारण अलग-अलग साहित्य, गीत और कहानियाँ होती हैं।

3. **ऐतिहासिक और राजनीतिक प्रभाव** भारत में विभिन्न राजवंश, उपनिवेश और राज्य थे। प्रत्येक राज्य में अपनी प्रशासनिक और सांस्कृतिक परंपराएँ थीं। इस कारण भोजन, त्योहार और रीति-रिवाज़ अलग-अलग हो गए।

4. **धर्म और त्योहार परंपरा** भारत में विभिन्न धर्म हैं, जैसे हिंदू, मुस्लिम, सिख, बौद्ध आदि। प्रत्येक धर्म का अपना त्योहार होता है और अलग-अलग रीति-रिवाज़ होते हैं।

5. **भोजन और त्योहार के कारण अलग-अलग संस्कृति** प्रत्येक राज्य में अलग-अलग भोजन और त्योहार होते हैं, जो उस राज्य की सांस्कृतिक पहचान है।

**भारतीय समाज पर सांस्कृतिक विविधता का सकारात्मक प्रभाव**

1. **सांस्कृतिक अभिव्यक्ति** विविधता के कारण भारत में विभिन्न कला, संगीत, नृत्य और साहित्य का विकास हुआ। यह भारत को एक सांस्कृतिक शक्ति बनाता है।

2. **आर्थिक प्रभाव** विविधता के कारण भारत में पर्यटन, हस्तशिल्प और स्थानीय बाजारों की विविधता है। यह आय और रोजगार का स्रोत बनता है।

3. **शिक्षा और अनुसंधान** विविधता के कारण भारत में विभिन्न भाषाओं, साहित्य, इतिहास और सामाजिक विज्ञान का अध्ययन होता है। यह भारतीय शिक्षा प्रणाली को मजबूत बनाता है।

4. **वैश्विक मंच पर पहचान** विविधता के कारण भारत में विभिन्न सांस्कृतिक उत्पाद वैश्विक मंच पर दिखाई देते हैं। यह भारत की छवि को मजबूत बनाता है।

5. **सामाजिक मूल्य और सहिष्णुता** विविधता के कारण भारत में विभिन्न धर्म, जातीय समूह और भाषाएँ हैं। यह सहिष्णुता और सामाजिक सहयोग का विकास करता है।

इस प्रकार, भारत की सांस्कृतिक विविधता एक सकारात्मक प्रभाव है, जो एकता के साथ समृद्धि और पहचान को विकसित करता है।

---

## 📊 Benchmarks

Our model has been evaluated across multiple benchmark categories to assess its capabilities in different domains. All results are reported as **zero-shot** performance unless otherwise specified.

---

### 🌐 General English Benchmarks

Performance on fundamental language understanding and reasoning tasks:

|      Task     | Param-2-17B-A2.4B | Qwen1.5-MoE-A2.7B | DeepSeek-R1-Distill-Qwen-14B | DeepSeek-V2-Lite | gpt-oss-20b |
| :-----------: | :---------------: | :---------------: | :--------------------------: | :--------------: | :---------: |
| ARC Challenge |       56.83       |       44.03       |             53.75            |       45.31      |    48.98    |
|    ARC Easy   |       77.69       |        68.9       |             74.75            |       70.41      |    75.51    |
|   HellaSwag   |       77.43       |       77.26       |             79.04            |       73.53      |     57.8    |
|    SQuADv2    |       33.86       |       13.96       |             23.77            |       25.2       |     8.66    |
|      MMLU     |       57.79       |       62.22       |             73.05            |       46.57      |    66.37    |

---

### 💻 Code, Math & Reasoning Benchmarks

Performance on specialized reasoning, mathematical problem-solving, and code generation:

|    Task   | Param-2-17B-A2.4B | Qwen1.5-MoE-A2.7B | DeepSeek-R1-Distill-Qwen-14B | DeepSeek-V2-Lite | gpt-oss-20b |
| :-------: | :---------------: | :---------------: | :--------------------------: | :--------------: | :---------: |
|   GSM8K   |       57.32       |       61.49       |             87.87            |       38.21      |    36.54    |
|   MathQA  |       40.23       |       35.61       |             48.44            |       38.73      |    36.68    |
| HumanEval |       36.59       |       33.54       |             26.22            |       26.83      |    24.39    |
|    MBPP   |        47         |       38.6        |             61               |       26.6       |      -      |

---

### 📝 Instruction Following Benchmarks

Evaluation of instruction comprehension and task execution capabilities:

|    Task    | Param-2-17B-A2.4B | Qwen1.5-MoE-A2.7B | DeepSeek-R1-Distill-Qwen-14B | DeepSeek-V2-Lite | gpt-oss-20b |
| :--------: | :---------------: | :---------------: | :--------------------------: | :--------------: | :---------: |
|  TriviaQA  |       54.64       |         38        |             17.88            |       34.62      |    29.05    |
|   LogiQA   |       52.35       |       31.18       |             33.49            |       29.34      |    32.87    |
| WinoGrande |       67.96       |       69.22       |             72.69            |       67.01      |    67.48    |
| TruthfulQA |       46.42       |       43.57       |             58.63            |       28.89      |    52.93    |
|   IFEval   |       34.05       |       13.19       |             51.8             |       12.75      |    31.42    |

---

### 🇮🇳 Indic Language Benchmarks

Performance on Indian language understanding and culturally grounded tasks:

|          Task         | Param-2-17B-A2.4B | Qwen1.5-MoE-A2.7B | DeepSeek-R1-Distill-Qwen-14B | DeepSeek-V2-Lite | gpt-oss-20b |
| :-------------------: | :---------------: | :---------------: | :--------------------------: | :--------------: | :---------: |
|       Sanskriti       |       66.54       |       28.71       |             74.58            |       64.84      |    75.14    |
| ARC Challenge (Indic) |       36.37       |       21.71       |             23.63            |       21.76      |    31.55    |
|  TriviaQA (Indic MCQ) |       72.95       |       25.21       |             53.99            |       29.18      |    71.67    |
|      Indic BoolQ      |       75.98       |       46.48       |             70.03            |       62.9       |    70.02    |
|      HellaSwag Hi     |       49.63       |       25.86       |             39.02            |       30.23      |    41.25    |
|        MMLU Hi        |       59.23       |       33.27       |             44.52            |       24.26      |    50.29    |

> **Note:** ARC Challenge (Indic) and TriviaQA (Indic MCQ) scores are **macro-averaged across 10 Indian languages**:  
> bn (Bengali), gu (Gujarati), hi (Hindi), kn (Kannada), ml (Malayalam), mr (Marathi), or (Odia), pa (Punjabi), ta (Tamil), te (Telugu).

---

> **Notes:**
> - All benchmarks reflect **zero-shot** performance unless noted
> - Evaluation pipeline follows standard LM Evaluation Harness protocols
> - All models have been tested with thinking=false
---

## ⚙️ Intended Use

This is an early post-training checkpoint is designed for:

- **Research and Development**: Investigating MoE architectures and multilingual language models
- **Downstream Fine-Tuning**: Building task-specific models for Indian language applications
- **Academic Research**: Studying cross-lingual transfer and low-resource language modeling
- **Application Development**: Creating culturally-aware AI applications for Indian markets
- **Community Innovation**: Enabling the open-source community to build India-specific use cases

### Recommended Use Cases

- Multilingual chatbots and conversational AI
- Content generation in Indian languages
- Cross-lingual information retrieval
- Educational tools for Indian language learning
- Document understanding and summarization
- Cultural and regional knowledge applications

---

## ⚠️ Limitations

### Technical Limitations

- **Hardware Requirements**: Despite sparse activation, the 17B total parameters require significant VRAM (approximately **34GB+ in bfloat16**) to load the full model
- **Context Window**: Optimized for 4096 tokens; performance may degrade on extremely long-form documents beyond this limit

### Model Limitations

- **No Guardrails**: The model has not undergone RLHF or other alignment procedures
- **Potential Biases**: May contain biases present in the training data
- **Language Imbalance**: Performance may vary across different Indian languages based on data availability
- **Not Production-Ready**: Requires fine-tuning and safety evaluation before production deployment

---

## 🔒 Responsible AI & Safety

### Important Considerations

1. **Base Model Status**: This is an early post-training checkpoint that requires additional fine-tuning and safety alignment before production use.

2. **Data Sources and Artifacts**: Training data includes content crawled from the internet, which may contain inherited artifacts, biases, or inaccuracies.

3. **No Safety Alignment**: The model has not undergone comprehensive safety training and may generate:
   - Biased or stereotypical content
   - Factually incorrect information
   - Potentially harmful or offensive text

4. **Deployment Requirements**: Organizations deploying this model must:
   - Implement appropriate content filtering and safety measures
   - Conduct thorough testing for their specific use case
   - Monitor outputs for quality and safety
   - Comply with applicable legal and ethical guidelines

5. **Community Responsibility**: Users are expected to:
   - Use the model ethically and responsibly
   - Avoid malicious applications
   - Report issues and contribute to model improvement
   - Credit the authors as per licensing terms

---

## 🤝 Acknowledgement & Collaboration

### A Home-Grown Effort

This work represents a **bottom-up initiative to develop large language models from scratch within India** with limited resources. It reflects our humble, resource-constrained journey to contribute meaningfully to the open-source AI ecosystem and foster collaboration within the broader community.

### Community Collaboration

We welcome contributions and open dialogue:
- Share feedback and report issues
- Collaborate on improvements and extensions
- Contribute to dataset curation and evaluation
- Build innovative applications on top of this foundation

Future versions will introduce better alignment, improved training scale, and more curated datasets. Together, we aim to evolve toward safer and more capable AI systems.

---

## 📜 License

This post-trained checkpoint is released under the **BharatGen non-commercial license**.

Please refer to the [LICENSE](./LICENSE) file for detailed terms and conditions.

---