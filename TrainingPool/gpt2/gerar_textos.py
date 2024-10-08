from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os

def gerar_texto(prompt, modelo_dir, tokenizer_dir=None, max_length=500, num_beams=10, temperature=0.7, top_p=0.9, top_k=50):
    # Verifica se o diretório do modelo existe
    if not os.path.exists(modelo_dir):
        raise ValueError(f"O diretório {modelo_dir} não existe.")

    # Se o diretório do tokenizer não for fornecido, usa o tokenizer padrão GPT-2
    if tokenizer_dir is None:
        print("Carregando o tokenizer padrão GPT-2.")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    else:
        print(f"Conteúdo do diretório do tokenizer {tokenizer_dir}: {os.listdir(tokenizer_dir)}")
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_dir)

    # Exibe o conteúdo do diretório do modelo para depuração
    print(f"Conteúdo do diretório do modelo {modelo_dir}: {os.listdir(modelo_dir)}")

    try:
        # Carrega o modelo do diretório especificado
        modelo = GPT2LMHeadModel.from_pretrained(modelo_dir)
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar o modelo: {e}")

    # Tokeniza o prompt de entrada
    entradas = tokenizer(prompt, return_tensors="pt")

    # Gera o texto a partir do modelo
    saida = modelo.generate(
        **entradas,
        max_length=max_length,
        num_beams=num_beams,            
        temperature=temperature,        
        top_p=top_p,                    
        top_k=top_k,                    
        num_return_sequences=1,
        no_repeat_ngram_size=2,         # Evita repetição de n-gramas para melhorar a qualidade
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id  # Define o token de padding
    )

    # Decodifica a saída gerada em texto
    texto_gerado = tokenizer.decode(saida[0], skip_special_tokens=True)
    return texto_gerado

if __name__ == "__main__":
    # Define o prompt inicial
    prompt_inicial = "Pergunte qualquer coisa..."

    # Diretório do modelo e do tokenizer
    caminho_modelo = os.path.abspath("caminho-modelo-treinado")

    # Gera o texto usando o modelo treinado
    texto = gerar_texto(prompt_inicial, caminho_modelo)
    print(texto)

    # Loop contínuo para permitir múltiplas perguntas
    while True:
        # Recebe o prompt do usuário
        prompt_inicial = input("Digite sua pergunta (ou 'sair' para encerrar): ")

        # Condição de saída
        if prompt_inicial.lower() in ['sair', 'exit']:
            print("Encerrando o programa...")
            break

        # Gera o texto com base na entrada do usuário
        texto = gerar_texto(prompt_inicial, caminho_modelo, max_length=500, num_beams=10, temperature=0.7, top_p=0.9, top_k=50)

        # Exibe o texto gerado
        print("\nTexto gerado:")
        print(texto)
        print("\n" + "-"*50 + "\n")
