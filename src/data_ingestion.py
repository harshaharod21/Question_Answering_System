#libraries to be imported
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import NLTKTextSplitter

from langchain_community.document_loaders import PyPDFLoader


def load_data():

    """ Loading data for three books from PDF"""

    # Example usage:
    file_paths = [
        "C:\All_projects\Question_Answering_System\Kafka on the Shore - Haruki Murakami [Worldfreebooks.com].pdf",
        "C:\All_projects\Question_Answering_System\Project Mary Hail.pdf",
        "C:\All_projects\Question_Answering_System\The Invisible Life of Addie LaRue By V E Schwab.pdf"
    ]
    pages = load_data(file_paths)
        
    pages = []

    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        pages.extend(loader.load_and_split())
    return pages





