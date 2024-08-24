import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset

def carregar_dataset(caminho_arquivo, tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    dataset = load_dataset('text', data_files={'train': caminho_arquivo}, split='train')
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=4)  # Aumentar o número de processos
    
    return tokenized_dataset

def preparar_dataset_para_treinamento(tokenized_dataset):
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    return tokenized_dataset

def treinar_modelo(dataset_path, output_dir):
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True  # Permitir TF32 para acelerar cálculos de matriz
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Especificar o diretório local onde os arquivos serão salvos
    local_model_dir = "./meu_modelo_local"

    # Garantir que o diretório existe
    os.makedirs(local_model_dir, exist_ok=True)
    
    # Carregar o tokenizador e o modelo, salvando no diretório local
    tokenizer = GPT2Tokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct')
    tokenizer.save_pretrained(local_model_dir)

    modelo = GPT2LMHeadModel.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct').to(device)
    modelo.save_pretrained(local_model_dir)

    # Carregar e preparar o dataset
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
        num_train_epochs=5,  # Aumentar o número de épocas
        per_device_train_batch_size=8,  # Ajustar o tamanho do batch
        gradient_accumulation_steps=4,
        learning_rate=5e-5,  # Aumentar a taxa de aprendizado
        save_steps=10_000,  # Aumentar a frequência de salvamento
        save_total_limit=2,
        logging_dir='./logs',
        logging_steps=500,  # Aumentar a frequência de logging
        eval_strategy="steps",
        eval_steps=2_500,  # Avaliação mais frequente
        fp16=True,  # Ativar mixed precision
        dataloader_num_workers=4,  # Aumentar o número de workers
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
