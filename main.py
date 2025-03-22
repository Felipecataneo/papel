import streamlit as st
import os
import tempfile
import io
from pdfminer.high_level import extract_text
import pandas as pd
import numpy as np
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
import re

st.set_page_config(page_title="Especialista em Óleo e Gás", layout="wide")

# Função para limpar e formatar texto extraído do PDF
def limpar_texto(texto):
    if not texto:
        return ""
    
    # Remover múltiplos espaços em branco
    texto = re.sub(r'\s+', ' ', texto)
    
    # Remover cabeçalhos/rodapés numéricos típicos de PDFs
    texto = re.sub(r'\n\d+\n', '\n', texto)
    
    # Remover marcadores de página
    texto = re.sub(r'Page \d+ of \d+', '', texto)
    
    # Remover caracteres não-imprimíveis
    texto = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', texto)
    
    # Converter quebras de linha consecutivas em parágrafos
    texto = re.sub(r'\n{3,}', '\n\n', texto)
    
    return texto.strip()

# Função para formatar texto em markdown
def formatar_markdown(texto):
    # Identificar possíveis títulos
    linhas = texto.split('\n')
    texto_formatado = []
    
    for i, linha in enumerate(linhas):
        linha = linha.strip()
        
        # Pular linhas vazias
        if not linha:
            texto_formatado.append('')
            continue
            
        # Detectar títulos potenciais (linhas curtas, menos de 80 caracteres)
        if len(linha) < 80 and linha.isupper():
            texto_formatado.append(f"## {linha.title()}")
        elif len(linha) < 80 and i > 0 and not linhas[i-1].strip():
            texto_formatado.append(f"### {linha}")
        # Transformar listas com marcadores numéricos ou traços
        elif re.match(r'^\d+\.\s', linha) or re.match(r'^-\s', linha):
            texto_formatado.append(linha)
        # Texto normal formata como parágrafo
        else:
            texto_formatado.append(linha)
    
    return '\n'.join(texto_formatado)

# Função para traduzir texto de inglês para português
def traduzir_texto(texto, client):
    if not texto:
        return ""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Você é um especialista em óleo e gás. Traduza o seguinte texto do inglês para o português, mantendo todos os termos técnicos precisos. Remova quaisquer ruídos ou artefatos da extração que não fazem parte do conteúdo principal do documento. Formate o texto em markdown adequadamente, identificando títulos, subtítulos, listas e parágrafos."},
                {"role": "user", "content": texto}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Erro ao traduzir texto: {e}")
        return texto

# Função para extrair texto de um PDF usando PyPDF2
def extrair_texto_pdf(arquivo_pdf):
    try:
        # Usando a função de alto nível do pdfminer que é mais atual e simples
        texto = extract_text(arquivo_pdf)
        return limpar_texto(texto)
    except Exception as e:
        st.error(f"Erro ao extrair texto do PDF: {e}")
        return ""

# Função para processar o documento
def processar_documento(arquivo, idioma, client):
    # Criar diretório temporário para armazenar o arquivo
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(arquivo.getvalue())
        temp_path = temp_file.name
    
    # Extrair texto do PDF
    texto = extrair_texto_pdf(temp_path)
    
    # Traduzir texto se estiver em inglês e o usuário quiser traduzi-lo
    if idioma == "Inglês":
        with st.spinner("Traduzindo documento..."):
            texto_final = traduzir_texto(texto, client)
    else:
        # Formatar o texto em português
        texto_final = formatar_markdown(texto)
    
    # Dividir texto em chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(texto_final)
    
    # Remover arquivo temporário
    os.unlink(temp_path)
    
    return texto_final, chunks

# Função para criar embeddings e vectorstore
def criar_vectorstore(chunks, api_key):
    embeddings = OpenAIEmbeddings(api_key=api_key)
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

# Função para configurar a cadeia de conversação
def configurar_cadeia_conversacao(vectorstore, api_key):
    llm = ChatOpenAI(
        api_key=api_key,
        model_name="gpt-4o-mini",
        temperature=0.2
    )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    cadeia_conversacao = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    
    return cadeia_conversacao

# Função para gerar resposta
def gerar_resposta(pergunta, cadeia_conversacao):
    resposta = ""
    if pergunta and cadeia_conversacao:
        with st.spinner("Gerando resposta..."):
            resultado = cadeia_conversacao.invoke({"question": pergunta})
            resposta = resultado["answer"]
    
    return resposta

# Interface principal
def main():
    st.title("Especialista em Óleo e Gás - Análise de Papers")
    
    # Barra lateral para configurações
    with st.sidebar:
        st.header("Configurações")
        
        # Campo para API Key da OpenAI
        api_key = st.text_input("OpenAI API Key", type="password")
        if api_key:
            st.session_state.openai_api_key = api_key
            os.environ["OPENAI_API_KEY"] = api_key
        
        # Upload do documento
        st.header("Upload de Documento")
        arquivo_pdf = st.file_uploader("Faça upload de um paper em PDF", type=["pdf"])
        
        # Seleção de idioma
        idioma = st.radio("Idioma do documento:", ["Português", "Inglês"])
        
        # Botão para processar
        processar = st.button("Processar Documento")
    
    # Inicialização de variáveis de sessão
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = None
    
    if "cadeia_conversacao" not in st.session_state:
        st.session_state.cadeia_conversacao = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    if "texto_final" not in st.session_state:
        st.session_state.texto_final = ""
    
    # Layout principal dividido em abas
    tab1, tab2 = st.tabs(["Documento", "Chat"])
    
    # Aba do documento
    with tab1:
        if st.session_state.texto_final:
            st.markdown(st.session_state.texto_final)
        else:
            st.info("Faça upload de um documento PDF e clique em 'Processar Documento' para visualizar o conteúdo.")
    
    # Aba de chat
    with tab2:
        st.subheader("Chat com o Especialista em Óleo e Gás")
        
        # Exibir histórico de chat
        for mensagem in st.session_state.chat_history:
            with st.chat_message(mensagem["role"]):
                st.markdown(mensagem["content"])
        
        # Campo de entrada para perguntas
        if pergunta := st.chat_input("Faça uma pergunta sobre o documento..."):
            # Adicionar pergunta ao histórico
            st.session_state.chat_history.append({"role": "user", "content": pergunta})
            
            # Exibir pergunta no chat
            with st.chat_message("user"):
                st.markdown(pergunta)
            
            # Verificar se temos o necessário para gerar uma resposta
            if not st.session_state.openai_api_key:
                with st.chat_message("assistant"):
                    st.warning("Por favor, insira sua OpenAI API Key na barra lateral.")
            elif not st.session_state.cadeia_conversacao:
                with st.chat_message("assistant"):
                    st.warning("Por favor, faça upload e processe um documento primeiro.")
            else:
                # Gerar e exibir resposta
                resposta = gerar_resposta(pergunta, st.session_state.cadeia_conversacao)
                
                # Adicionar resposta ao histórico
                st.session_state.chat_history.append({"role": "assistant", "content": resposta})
                
                # Exibir resposta no chat
                with st.chat_message("assistant"):
                    st.markdown(resposta)
    
    # Processamento do documento quando o botão é clicado
    if processar and arquivo_pdf is not None:
        if not st.session_state.openai_api_key:
            st.sidebar.warning("Por favor, insira sua OpenAI API Key primeiro.")
        else:
            try:
                # Inicializar cliente OpenAI
                client = OpenAI(api_key=st.session_state.openai_api_key)
                
                # Processar documento
                with st.spinner("Processando documento..."):
                    texto_final, chunks = processar_documento(arquivo_pdf, idioma, client)
                    
                    # Armazenar apenas o texto final na sessão
                    st.session_state.texto_final = texto_final
                
                # Criar vectorstore
                with st.spinner("Criando embeddings..."):
                    vectorstore = criar_vectorstore(chunks, st.session_state.openai_api_key)
                
                # Configurar cadeia de conversação
                st.session_state.cadeia_conversacao = configurar_cadeia_conversacao(vectorstore, st.session_state.openai_api_key)
                
                st.sidebar.success("Documento processado com sucesso! Navegue pelas abas para visualizar o conteúdo e fazer perguntas.")
                
                # Mostra a aba de documento automaticamente
                st.experimental_rerun()
            
            except Exception as e:
                st.sidebar.error(f"Erro ao processar documento: {e}")

if __name__ == "__main__":
    main()