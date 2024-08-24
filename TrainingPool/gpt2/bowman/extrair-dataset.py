import os
import PyPDF2

def extrair_texto_pdf(pasta_pdf):
    textos = []
    for arquivo in os.listdir(pasta_pdf):
        if arquivo.endswith(".pdf"):
            caminho = os.path.join(pasta_pdf, arquivo)
            with open(caminho, "rb") as f:
                leitor_pdf = PyPDF2.PdfReader(f)
                texto = ""
                for pagina in leitor_pdf.pages:
                    texto += pagina.extract_text()
                textos.append(texto)

    return textos

if __name__ == "__main__":
    textos = extrair_texto_pdf("pdfs")
    with open("dataset.txt", "w") as f:
        f.write("\n".join(textos))
        