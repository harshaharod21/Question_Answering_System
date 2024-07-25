from langchain_text_splitters import NLTKTextSplitter

def preprocess_text(pages):
    """Combining the text and splitting by 100 tokens approx.Maintaining boundaries using NLTK Text splitter"""


    # for experiment
    first_10= pages[:7]

    combined_text = ''.join(page.page_content for page in first_10)


    #use nltk to split the pages document to chunks of size 100
    first_10= pages[:7]

    text_splitter = NLTKTextSplitter(chunk_size=650,chunk_overlap=0)
    texts = text_splitter.split_text(combined_text)

    return texts




