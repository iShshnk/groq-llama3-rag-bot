# GROQ and LLAMA-3 Custom Retrieval Augmented Generation Bot for your Documents via Gradio

This project implements a powerful Retrieval Augmented Generation (RAG) system using GROQ and LLAMA-3-8B model, with a user-friendly Gradio interface. It allows users to upload PDF documents, process them, and ask questions about their content, leveraging advanced language models to provide accurate answers.

![Gradio Interface](https://github.com/iShshnk/groq-llama3-rag-bot/blob/main/media/A.png)

## Features

- PDF document upload and processing
- Question answering based on the content of the uploaded PDF
- User-friendly interface powered by Gradio
- Utilizes GROQ and LLAMA-3 and Langchain for natural language processing
- Document chunking and embedding storage is provided via ChromaDB Vector Database for efficient search and retrieval 

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.7+
- Groq API key

## Getting a Groq API Key

To use this application, you'll need a Groq API key. Follow these steps to obtain one:

1. Go to the [Groq website](https://www.groq.com/) and sign up for an account if you haven't already.
2. Once logged in, navigate to the API section of your account dashboard.
3. Generate a new API key. Make sure to copy it immediately, as you might not be able to see it again.
4. Keep this API key secure and do not share it publicly.

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/iShshnk/groq-llama3-rag-bot.git
   ```
   ```
   cd groq-llama3-rag-bot
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your Groq API key:
   - Open `app.py`
   - Replace `"your_groq_api_key_here"` with your actual Groq API key

## Usage

1. Run the application:
   ```
   python app.py
   ```

2. Open your web browser and go to the URL provided in the terminal (usually `http://127.0.0.1:7860`).

3. Use the Gradio interface to interact with the system:

   Step 1: Upload PDF
   ![Upload PDF]
   (https://github.com/iShshnk/groq-llama3-rag-bot/blob/main/media/1.png)

   Step 2: Process the PDF
   ![Process PDF](https://github.com/iShshnk/groq-llama3-rag-bot/blob/main/media/2.png)

   Step 3: Enter your Question
   ![Enter Question](https://github.com/iShshnk/groq-llama3-rag-bot/blob/main/media/3.png)

   Step 4: Get Answer
   ![Get Answer](https://github.com/iShshnk/groq-llama3-rag-bot/blob/main/media/4.png)

## How It Works

1. **PDF Upload**: Use the file upload component in the Gradio interface to select and upload your PDF document.

2. **PDF Processing**: After uploading, click the "Process PDF" button. This will:
   - Extract text from the PDF
   - Split the text into chunks
   - Create embeddings for each chunk
   - Store the embeddings in a vector database for quick retrieval

3. **Asking Questions**: Once the PDF is processed, you can type your questions into the text input field and click "Submit" to get answers.

4. **Retrieving Answers**: The system will:
   - Convert your question into an embedding
   - Search the vector database for relevant chunks
   - Use GROQ and LLAMA-3 models to generate an answer based on the retrieved chunks and your question

## Customization

You can customize this project by:
- Adjusting the chunk size for document splitting
- Experimenting with different embedding models
- Fine-tuning the GROQ and LLAMA-3 models for your specific use case

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Groq](https://console.groq.com/keys) for providing the LLM API
- [LangChain](https://github.com/hwchase17/langchain) for the document processing and QA chain
- [Gradio](https://gradio.app/) for the user interface framework

© 2024 Shashank Ramesh
