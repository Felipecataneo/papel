import streamlit as st
import os
import tempfile
import io
import uuid
from pdfminer.high_level import extract_text
import pandas as pd
import numpy as np
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
import re
import tiktoken  # Para contagem de tokens

st.set_page_config(page_title="Especialista em Óleo e Gás", layout="wide")

# Modelos atualizados para melhor qualidade
EMBEDDING_MODEL = "text-embedding-3-small"  # Modelo mais recente de embeddings
CHAT_MODEL = "gpt-4o"  # Modelo mais avançado para respostas

# Função para contar tokens
def num_tokens(text: str, model: str = CHAT_MODEL) -> int:
    """Retorna o número de tokens em um texto usando o mesmo tokenizador do modelo"""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

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
            model=CHAT_MODEL,  # Usando o modelo mais avançado para tradução
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

# Função para extrair texto de um PDF usando pdfminer
def extrair_texto_pdf(arquivo_pdf):
    try:
        # Usando a função de alto nível do pdfminer que é mais atual e simples
        texto = extract_text(arquivo_pdf)
        return limpar_texto(texto)
    except Exception as e:
        st.error(f"Erro ao extrair texto do PDF: {e}")
        return ""

# Melhorada a divisão de texto para melhor preservar contexto semântico
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
    
    # Dividir texto em chunks com melhor preservação semântica
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Chunks menores para maior precisão
        chunk_overlap=100,  # Sobreposição para manter contexto
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]  # Prioridade de separação
    )
    chunks = text_splitter.split_text(texto_final)
    
    # Remover arquivo temporário
    os.unlink(temp_path)
    
    return texto_final, chunks

# Função aprimorada para criar embeddings e vectorstore
def criar_vectorstore(chunks, api_key):
    if not chunks:
        st.error("Não há chunks para processar. O documento pode estar vazio.")
        return None
        
    try:
        # Usando o modelo de embeddings mais recente da OpenAI
        embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=api_key,
            dimensions=1536  # Dimensionalidade explícita
        )
        
        # Criar metadata para cada chunk para melhor rastreabilidade
        metadatas = [{"chunk_id": i, "token_count": num_tokens(chunk)} for i, chunk in enumerate(chunks)]
        
        # Criar vectorstore com metadados
        vectorstore = FAISS.from_texts(
            texts=chunks, 
            embedding=embeddings,
            metadatas=metadatas
        )
        
        return vectorstore
    except Exception as e:
        st.error(f"Erro ao criar vectorstore: {e}")
        return None

# Função aprimorada para configurar o grafo de conversação com retrieval aumentado
def configurar_grafo_conversacao(vectorstore, api_key):
    if vectorstore is None:
        st.error("VectorStore não foi criado corretamente!")
        return None
    
    # Definir o modelo de chat com modelo mais poderoso
    model = ChatOpenAI(
        api_key=api_key,
        model_name=CHAT_MODEL,
        temperature=0.2
    )
    
    # Definir o grafo de estado
    workflow = StateGraph(state_schema=MessagesState)
    
    # Função para chamar o modelo com contexto do documento e retrieval aprimorado
    def call_model_with_context(state: MessagesState):
        # Armazenar referência ao vectorstore que está no escopo externo
        nonlocal vectorstore
        
        # Obter a pergunta mais recente
        latest_message = state["messages"][-1]
        
        if isinstance(latest_message, HumanMessage):
            try:
                # Extrair a pergunta do usuário
                query = latest_message.content
                
                # Reformular a consulta para melhorar a recuperação (query expansion)
                client = OpenAI(api_key=api_key)
                query_expansion_response = client.chat.completions.create(
                    model="gpt-4o-mini",  # Modelo mais leve para reformulação
                    messages=[
                        {"role": "system", "content": "Você é um assistente especializado em expandir consultas para melhorar a recuperação de informações. Reformule a pergunta do usuário para incluir termos relacionados e sinônimos que possam ajudar na busca em embeddings, mantendo a intenção original. Forneça apenas a consulta reformulada, sem explicações adicionais."},
                        {"role": "user", "content": f"Reformule esta consulta para busca em um documento técnico sobre óleo e gás: '{query}'"}
                    ],
                    temperature=0.2,
                    max_tokens=200
                )
                
                expanded_query = query_expansion_response.choices[0].message.content
                
                # Fazer múltiplas pesquisas com diferentes estratégias e combinar resultados
                # 1. Pesquisa com consulta original
                docs_original = vectorstore.similarity_search(query, k=3)
                
                # 2. Pesquisa com consulta expandida
                docs_expanded = vectorstore.similarity_search(expanded_query, k=3)
                
                # 3. Pesquisa MMR (Maximum Marginal Relevance) para diversidade
                docs_mmr = vectorstore.max_marginal_relevance_search(query, k=2, fetch_k=5)
                
                # Combinar resultados, removendo duplicatas
                all_docs = []
                seen_content = set()
                
                for doc_set in [docs_original, docs_expanded, docs_mmr]:
                    for doc in doc_set:
                        if doc.page_content not in seen_content:
                            all_docs.append(doc)
                            seen_content.add(doc.page_content)
                
                # Limitar a no máximo 6 documentos para não sobrecarregar o contexto
                unique_docs = all_docs[:6]
                
                if not unique_docs:
                    context = "Não foi possível encontrar informações relevantes no documento."
                else:
                    # Ordenar por relevância (assumindo que docs_original tem a ordem mais relevante)
                    # Criar um contexto com os documentos relevantes e seus metadados
                    context_parts = []
                    for i, doc in enumerate(unique_docs):
                        # Adicionar metadados se disponíveis
                        metadata_str = ""
                        if hasattr(doc, 'metadata') and doc.metadata:
                            metadata_str = f" [Chunk ID: {doc.metadata.get('chunk_id', 'N/A')}]"
                        
                        context_parts.append(f"TRECHO {i+1}{metadata_str}:\n{doc.page_content}")
                    
                    context = "\n\n".join(context_parts)
                
                # Calcular tokens do contexto para evitar estourar o limite
                context_tokens = num_tokens(context)
                
                # Se o contexto for muito grande, reduza-o
                if context_tokens > 3000:  # Limite conservador para deixar espaço para o resto do prompt
                    # Encurtar o contexto mantendo os trechos mais relevantes
                    unique_docs = unique_docs[:3]  # Reduzir para os 3 mais relevantes
                    context_parts = [f"TRECHO {i+1}:\n{doc.page_content}" for i, doc in enumerate(unique_docs)]
                    context = "\n\n".join(context_parts)
                
                # Criar uma mensagem de sistema com o contexto mais sofisticada
                system_message = f"""Você é um especialista em óleo e gás respondendo perguntas sobre um documento técnico.
                
                INSTRUÇÕES:
                1. Use APENAS as informações fornecidas nos trechos abaixo para responder à pergunta.
                2. Se a informação necessária não estiver nos trechos, diga claramente "Não encontrei essa informação específica no documento".
                3. Cite trechos específicos do documento para apoiar sua resposta.
                4. Seja conciso e direto em suas respostas.
                5. Não invente informações que não estejam nos trechos fornecidos.
                6. Formate sua resposta em markdown quando apropriado.
                
                PERGUNTA ORIGINAL: {query}
                
                CONTEXTO DO DOCUMENTO:
                {context}
                """
                
                # Preparar as mensagens para o modelo com contexto completo
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": query}
                ]
                
                # Adicionar histórico de chat limitado para contexto da conversa
                # Obter até 4 mensagens anteriores (2 pares de pergunta-resposta)
                chat_history = []
                full_history = state["messages"]
                
                if len(full_history) > 1:  # Se há mais de uma mensagem no histórico
                    # Adicionar até 4 mensagens anteriores, omitindo a mais recente (que já estamos processando)
                    history_to_add = full_history[-5:-1] if len(full_history) > 5 else full_history[:-1]
                    
                    for msg in history_to_add:
                        if isinstance(msg, HumanMessage):
                            chat_history.append({"role": "user", "content": msg.content})
                        elif isinstance(msg, AIMessage):
                            chat_history.append({"role": "assistant", "content": msg.content})
                
                # Inserir histórico entre sistema e usuário atual, se houver
                if chat_history:
                    messages = [messages[0]] + chat_history + [messages[1]]
                
                # Chamar o modelo com o contexto completo
                response = model.invoke(messages)
                
                # Retornar a resposta para adicionar ao estado
                return {"messages": [AIMessage(content=response.content)]}
                
            except Exception as e:
                error_msg = f"Erro ao processar sua pergunta: {str(e)}"
                return {"messages": [AIMessage(content=error_msg)]}
        
        return {"messages": []}
    
    # Adicionar nó e borda ao grafo
    workflow.add_node("model", call_model_with_context)
    workflow.add_edge(START, "model")
    
    # Configurar a memória
    memory = MemorySaver()
    
    # Compilar o grafo
    app = workflow.compile(checkpointer=memory)
    
    return app

# Função para gerar resposta usando o grafo LangGraph (mantida como estava)
def gerar_resposta_langgraph(pergunta, app, thread_id):
    resposta = ""
    if not pergunta:
        return "Por favor, faça uma pergunta."
        
    if not app:
        return "O sistema de chat não foi inicializado corretamente. Verifique se o documento foi processado."
        
    with st.spinner("Gerando resposta..."):
        try:
            # Configurar o thread_id para esta conversa
            config = {"configurable": {"thread_id": thread_id}}
            
            # Criar mensagem humana
            input_message = HumanMessage(content=pergunta)
            
            # Obter resposta do grafo
            for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
                if event.get("messages") and len(event["messages"]) > 0:
                    resposta = event["messages"][-1].content
                    
            if not resposta:
                resposta = "Não foi possível gerar uma resposta. Pode haver um problema com a integração do modelo."
                
        except Exception as e:
            resposta = f"Erro ao gerar resposta: {str(e)}"
            st.error(resposta)
    
    return resposta

# Interface principal (o restante do código permanece igual ao original)
def main():
    st.title("Especialista em Óleo e Gás - Análise de Papers")
    
    # Inicialização de variáveis de sessão
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = None
    
    if "langgraph_app" not in st.session_state:
        st.session_state.langgraph_app = None
    
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    if "texto_final" not in st.session_state:
        st.session_state.texto_final = ""
        
    if "documento_processado" not in st.session_state:
        st.session_state.documento_processado = False
        
    if "tab_ativa" not in st.session_state:
        st.session_state.tab_ativa = "Documento"
    
    if "processando" not in st.session_state:
        st.session_state.processando = False
        
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    
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
        
        # Seleção de modelos (adicionando opção para o usuário escolher)
        modelo_chat = st.selectbox(
            "Modelo de Chat:",
            ["gpt-4o", "gpt-4o-mini"],
            index=0
        )
        
        # Atualizar constante global com base na seleção do usuário
        global CHAT_MODEL
        CHAT_MODEL = modelo_chat
        
        # Botão para processar
        processar = st.button("Processar Documento")
        
        # Verificação após processamento - mais simplificada
        if st.session_state.documento_processado:
            st.success("Documento processado com sucesso!")
    
    # Layout principal dividido em abas
    tabs = st.tabs(["Documento", "Chat"])
    
    # Selecionar a aba ativa
    tab_index = 0 if st.session_state.tab_ativa == "Documento" else 1
    
    # Aba do documento
    with tabs[0]:
        if st.session_state.processando:
            st.info("Processando documento... Por favor, aguarde.")
            st.spinner()
        elif st.session_state.texto_final:
            # Adicionar botão para copiar o texto
            col1, col2 = st.columns([1, 6])
            with col1:
                if st.button("📋 Copiar texto", key="copy_button"):
                    st.session_state["texto_copiado"] = True
                    # Usamos clipboard para mostrar ao usuário que tentamos copiar
                    st.success("Texto copiado! Use Ctrl+V para colar.")
            
            # Adicionar campo de texto escondido para possibilitar a cópia manual
            with st.expander("Se o botão de cópia não funcionar, selecione e copie o texto abaixo:", expanded=st.session_state.get("texto_copiado", False)):
                st.code(st.session_state.texto_final, language="markdown")
            
            # Exibir o documento processado
            st.markdown(st.session_state.texto_final)
        else:
            st.info("Faça upload de um documento PDF e clique em 'Processar Documento' para visualizar o conteúdo.")
    
    # Aba de chat
    with tabs[1]:
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
            elif not st.session_state.langgraph_app or not st.session_state.vectorstore:
                with st.chat_message("assistant"):
                    st.warning("Por favor, faça upload e processe um documento primeiro, ou houve um erro no processamento do documento.")
            else:
                # Gerar e exibir resposta usando LangGraph
                resposta = gerar_resposta_langgraph(
                    pergunta, 
                    st.session_state.langgraph_app, 
                    st.session_state.thread_id
                )
                
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
                # Atualizar estado para indicar processamento
                st.session_state.processando = True
                st.session_state.documento_processado = False
                st.session_state.texto_final = ""
                st.session_state.vectorstore = None
                st.session_state.texto_copiado = False  # Resetar estado de cópia
                
                # Forçar a atualização da interface para mostrar o estado de processamento
                st.rerun()
                
            except Exception as e:
                st.sidebar.error(f"Erro ao iniciar processamento: {e}")
                st.session_state.processando = False

    # Lógica de processamento separada para garantir que a interface seja atualizada corretamente
    if st.session_state.processando and arquivo_pdf is not None:
        try:
            # Inicializar cliente OpenAI
            client = OpenAI(api_key=st.session_state.openai_api_key)
            
            # Processar documento
            texto_final, chunks = processar_documento(arquivo_pdf, idioma, client)
            
            # Verificação básica de chunks sem mensagens de debug
            if not chunks:
                st.sidebar.error("Nenhum conteúdo foi extraído do documento!")
                st.session_state.processando = False
                st.rerun()
                return
            
            # Criar vectorstore - sem mensagens de debug
            with st.spinner("Criando índice de busca..."):
                vectorstore = criar_vectorstore(chunks, st.session_state.openai_api_key)
            
            # Salvar o vectorstore na sessão
            st.session_state.vectorstore = vectorstore
            
            if not vectorstore:
                st.error("Não foi possível criar o índice de busca. O chat pode não funcionar corretamente.")
                st.session_state.processando = False
                st.rerun()
                return
            
            # Configurar o grafo de conversação com LangGraph
            app = configurar_grafo_conversacao(vectorstore, st.session_state.openai_api_key)
            
            if not app:
                st.error("Não foi possível configurar o sistema de chat.")
                st.session_state.processando = False
                st.rerun()
                return
            
            # Atualizar o estado da sessão
            st.session_state.texto_final = texto_final
            st.session_state.langgraph_app = app
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.chat_history = []
            st.session_state.documento_processado = True
            st.session_state.processando = False
            st.session_state.tab_ativa = "Documento"
            
            # Forçar atualização da interface
            st.rerun()
            
        except Exception as e:
            st.sidebar.error(f"Erro ao processar documento: {e}")
            st.session_state.processando = False
            st.rerun()

if __name__ == "__main__":
    main()