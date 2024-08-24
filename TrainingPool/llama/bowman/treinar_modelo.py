import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset
from huggingface_hub import login

def carregar_dataset(caminho_arquivo, tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    dataset = load_dataset('text', data_files={'train': caminho_arquivo}, split='train')
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=4)
    
    return tokenized_dataset

def preparar_dataset_para_treinamento(tokenized_dataset):
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    return tokenized_dataset

def treinar_modelo(dataset_path, output_dir):
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    local_model_dir = "./meu_modelo_local"
    os.makedirs(local_model_dir, exist_ok=True)
    
    # Configurar autenticação com o token do Hugging Face
    huggingface_token = "SEU_TOKEN_DE_ACESSO"  # Substitua pelo seu token de acesso
    login(token=huggingface_token)

    # Carregar o tokenizer e o modelo com autenticação
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct', use_auth_token=huggingface_token)
    tokenizer.save_pretrained(local_model_dir)

    modelo = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct', use_auth_token=huggingface_token).to(device)
    modelo.save_pretrained(local_model_dir)

    dataset = carregar_dataset(dataset_path, tokenizer)
    dataset = preparar_dataset_para_treinamento(dataset)

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt"
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,  # Ajustado para lidar com o tamanho do modelo
        gradient_accumulation_steps=8,  # Aumentado para compensar o batch menor
        learning_rate=5e-5,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir='./logs',
        logging_steps=500,
        evaluation_strategy="steps",
        eval_steps=2_500,
        fp16=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        max_grad_norm=1.0,
    )

    treinador = Trainer(
        model=modelo,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    treinador.train()
    treinador.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    treinar_modelo("dataset.txt", "modelo_treinado")
