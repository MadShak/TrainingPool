# TrainingPool

**TrainingPool** é uma aplicação projetada para treinar e executar modelos Transformer. Este repositório contém scripts para treinar modelos baseados no GPT-2 e Llama-3.1, bem como extrair texto de PDFs e criar datasets para o treinamento.

## Funcionalidades

- **Treinamento de Modelos**: Scripts para treinar modelos Transformer usando datasets personalizados.
- **Extração de Texto de PDFs**: Ferramenta para converter PDFs em texto para criação de datasets.
- **Geração de Texto**: Interface para gerar texto usando um modelo treinado com base em prompts de entrada.

## Estrutura do Repositório

- `extrair_texto_pdf.py`: Script para extrair texto de arquivos PDF e salvar em um arquivo `.txt`.
- `treinar_modelo.py`: Script para treinar um modelo Transformer com base em um dataset fornecido.
- `gerar_texto.py`: Script para gerar texto a partir de um prompt usando um modelo pré-treinado.
- `requirements.txt`: Lista de dependências necessárias para executar os scripts.

## Instalação

1. Clone este repositório para a sua máquina local:

   git clone https://github.com/MadShak/TrainingPool.git

2. Navegue até o diretório do projeto:

   cd TrainingPool

3. Crie e ative um ambiente virtual (opcional, mas recomendado):

   python3 -m venv venv
   source venv/bin/activate  # Linux/MacOS
   venv\Scripts\activate  # Windows

4. Instale as dependências:

   pip install -r requirements.txt

## Uso

### Extração de Texto de PDFs

Coloque seus arquivos PDF em um diretório chamado `pdfs` e execute o script de extração:

python extrair_texto_pdf.py

Isso criará um arquivo `dataset.txt` com o texto extraído dos PDFs.

### Treinamento do Modelo

Após a criação do dataset, treine seu modelo:

python treinar_modelo.py

Isso treinará o modelo com base nos parâmetros definidos no script.

### Geração de Texto

Depois de treinar o modelo, você pode gerar texto com base em um prompt:

python gerar_texto.py

Siga as instruções na interface para fornecer seu prompt e gerar o texto.

## Licença

Este projeto está licenciado sob a licença MIT - consulte o arquivo [LICENSE](LICENSE) para obter mais detalhes.

### `requirements.txt`

torch==2.0.1
transformers==4.31.0
datasets==2.14.0
PyPDF2==3.0.1
