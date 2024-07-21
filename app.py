import os
import tempfile
import gradio as gr
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.llms.base import LLM
from groq import Groq
from typing import Any, List, Optional

# Set up Groq client
GROQ_API_KEY = "your_groq_api_key_here"
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
groq_client = Groq(api_key=GROQ_API_KEY)

# Custom LLM class for Groq
class GroqLLM(LLM):
    client: Any
    model: str = "llama3-8b-8192"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        chat_completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
        )
        return chat_completion.choices[0].message.content

    @property
    def _llm_type(self) -> str:
        return "groq"

# Initialize GroqLLM
llm = GroqLLM(client=groq_client)

# Custom prompt template
template = """You are a direct and concise assistant. Answer the question using only the information provided in the context. Give only the specific answer requested, with no additional explanation or information.

Context: {context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(
    template=template, input_variables=["context", "question"]
)

class PDFQuestionAnswering:
    def __init__(self):
        self.qa_system = None

    def setup_qa_system(self, pdf_file):
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_file)
            temp_file_path = temp_file.name

        # Load PDF
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()

        # Remove the temporary file
        os.unlink(temp_file_path)

        # Text splitting
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)

        # Embeddings
        embeddings = HuggingFaceEmbeddings()

        # Vector store
        docsearch = Chroma.from_documents(texts, embeddings)

        # Set up RetrievalQA
        self.qa_system = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=docsearch.as_retriever(),
            chain_type_kwargs={"prompt": PROMPT}
        )

        return "PDF processed successfully. You can now ask questions."

    def get_answer(self, question):
        if self.qa_system is None:
            return "Please upload a PDF file first."

        raw_answer = self.qa_system.run(question)
        return raw_answer.strip()

pdf_qa = PDFQuestionAnswering()

def process_pdf(pdf_file):
    if pdf_file is None:
        return "Please upload a PDF file."
    return pdf_qa.setup_qa_system(pdf_file)

def answer_question(question):
    return pdf_qa.get_answer(question)

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# GROQ and LLAMA-3 Custom RAG Bot ")
    with gr.Row():
        pdf_input = gr.File(label="Upload PDF", type="binary", file_types=[".pdf"])
        pdf_output = gr.Textbox(label="PDF Processing Status")
    pdf_button = gr.Button("Process PDF")
    
    with gr.Row():
        question_input = gr.Textbox(label="Enter your question")
        answer_output = gr.Textbox(label="Answer")
    question_button = gr.Button("Get Answer")

    pdf_button.click(process_pdf, inputs=[pdf_input], outputs=[pdf_output])
    question_button.click(answer_question, inputs=[question_input], outputs=[answer_output])

if __name__ == "__main__":
    demo.launch()
