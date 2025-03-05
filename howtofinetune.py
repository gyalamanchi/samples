pip install transformers peft bitsandbytes datasets accelerate trl


#Load dataset
import json

data = [
    {"condition": "Hypertension", "symptoms": "Headache, dizziness, chest pain", "treatment": "Lifestyle changes, medication"},
    {"condition": "Diabetes", "symptoms": "Increased thirst, frequent urination, fatigue", "treatment": "Insulin, diet control, exercise"},
    {"condition": "Asthma", "symptoms": "Wheezing, coughing, shortness of breath", "treatment": "Inhalers, medication"},
    {"condition": "Arthritis", "symptoms": "Joint pain, stiffness, swelling", "treatment": "Pain relievers, physical therapy"},
    {"condition": "Migraine", "symptoms": "Severe headache, nausea, sensitivity to light", "treatment": "Pain medication, rest"},
]

with open("medical_conditions.json", "w") as f:
    json.dump(data, f, indent=4)




#Prepare dataset for training
from datasets import load_dataset

dataset = load_dataset("json", data_files="medical_conditions.json")

def format_prompt(example):
    prompt = f"Condition: {example['condition']}\nSymptoms: {example['symptoms']}\nTreatment: {example['treatment']}\n\n"
    return {"text": prompt}

dataset = dataset.map(format_prompt)



#Load the LLM and Configure LoRA
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

model_name = "meta-llama/Llama-2-7b-chat-hf" # or your chosen model.
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model.config.use_cache = False
model.config.pretraining_tp = 1
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)



#Configure training arguments and trainer
training_args = TrainingArguments(
    output_dir="./medical_finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    report_to="tensorboard",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    peft_config=lora_config,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_args,
)


#train the model
trainer.train()
model.save_pretrained("./medical_finetuned")



#use in FineTuned:
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, "./medical_finetuned")

prompt = "Condition: Diabetes\nSymptoms:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
