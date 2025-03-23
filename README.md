# Especialista em Óleo e Gás - Análise de Papers

Uma aplicação Streamlit que permite processar documentos técnicos sobre óleo e gás, traduzir conteúdo em inglês para português e criar um chatbot especializado para consultar informações presentes no documento.

## Visão Geral

Esta aplicação foi desenvolvida para facilitar a leitura e consulta de documentos técnicos na área de óleo e gás. Ela oferece as seguintes funcionalidades:

- Upload de documentos PDF
- Extração de texto com remoção de ruídos
- Tradução automática de documentos em inglês para português
- Formatação em Markdown para melhor legibilidade
- Interface de chat para consultar informações contidas no documento

## Pré-requisitos

- Python 3.8 ou superior
- Conta na OpenAI com API key válida

## Instalação

1. Clone este repositório:
   ```bash
   git clone https://seu-repositorio/especialista-oleo-gas.git
   cd especialista-oleo-gas
   ```

2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

3. Inclua as seguintes dependências no seu arquivo `requirements.txt`:
   ```
   aiohttp==3.11.14
altair==5.5.0
faiss-cpu==1.10.0
GitPython==3.1.44
httpx==0.28.1
httpx-sse==0.4.0
langchain==0.3.21
langchain-community==0.3.20
langchain-core==0.3.47
langchain-openai==0.3.9
langchain-text-splitters==0.3.7
langgraph==0.3.18
langgraph-checkpoint==2.0.21
langgraph-prebuilt==0.1.4
langgraph-sdk==0.1.58
langsmith==0.3.18
narwhals==1.31.0
numpy==2.2.4
openai==1.68.2
pandas==2.2.3
pdfminer.six==20240706
pillow==11.1.0
pydantic==2.10.6
pydantic-settings==2.8.1
python-dotenv==1.0.1
regex==2024.11.6
requests==2.32.3
streamlit==1.43.2
tiktoken==0.9.0
   ```

## Uso

1. Inicie a aplicação:
   ```bash
   streamlit run app.py
   ```

2. No navegador, a interface será carregada automaticamente ou acesse: `http://localhost:8501`

3. Na barra lateral:
   - Insira sua API Key da OpenAI
   - Faça upload de um arquivo PDF
   - Selecione o idioma do documento (Português ou Inglês)
   - Clique em "Processar Documento"

4. Navegue entre as abas:
   - **Documento**: visualize o texto extraído do PDF (em português)
   - **Chat**: faça perguntas específicas sobre o conteúdo do documento

## Funcionamento

### Processamento de Documentos
1. O texto é extraído do PDF usando a biblioteca pdfminer.six
2. Se o documento estiver em inglês, ele é traduzido usando a API da OpenAI
3. O texto é limpo para remover ruídos comuns da extração de PDFs
4. O conteúdo é formatado em Markdown para melhor visualização

### Chatbot Especializado
1. O texto é dividido em pequenos segmentos (chunks)
2. Os chunks são transformados em embeddings usando a OpenAI
3. Os embeddings são armazenados em um banco de vetores FAISS
4. O chatbot usa busca semântica para encontrar as partes relevantes do documento
5. O modelo GPT da OpenAI gera respostas baseadas no contexto encontrado

## Estrutura do Código

- `extrair_texto_pdf()`: Extrai texto de arquivos PDF
- `limpar_texto()`: Remove ruídos e formata o texto extraído
- `formatar_markdown()`: Converte o texto para formato Markdown legível
- `traduzir_texto()`: Traduz documentos em inglês para português
- `processar_documento()`: Coordena o processo de extração e preparação do documento
- `criar_vectorstore()`: Cria o banco de dados de vetores para busca semântica
- `configurar_cadeia_conversacao()`: Configura o pipeline de conversação do LangChain
- `gerar_resposta()`: Processa perguntas e gera respostas baseadas no documento

## Limitações

- O processamento de documentos muito grandes pode ser lento
- A qualidade da extração pode variar dependendo do formato do PDF
- É necessária uma conexão com a Internet para tradução e consultas
- A aplicação requer uma API key válida da OpenAI com créditos suficientes

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para enviar pull requests ou abrir issues para melhorias na aplicação.

## Licença

[MIT](LICENSE)