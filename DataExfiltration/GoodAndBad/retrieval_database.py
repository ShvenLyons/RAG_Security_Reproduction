import os
import json
import torch
import time
from tqdm import tqdm
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import TextSplitter, RecursiveCharacterTextSplitter

def find_all_file(path):
    for root, ds, fs in os.walk(path):
        for f in fs:
            yield os.path.join(root, f)

def get_encoding_of_file(path):
    from chardet.universaldetector import UniversalDetector
    detector = UniversalDetector()
    with open(path, 'rb') as file:
        for line in file:
            detector.feed(line)
            if detector.done:
                break
    detector.close()
    return detector.result['encoding']

class LineBreakTextSplitter(TextSplitter):
    def split_text(self, text: str):
        return text.split("\n\n")

def get_text_splitter(strategy='linebreak', **kwargs):
    if strategy == 'linebreak':
        return LineBreakTextSplitter()
    elif strategy == 'recursive':
        return RecursiveCharacterTextSplitter(
            chunk_size=kwargs.get('chunk_size', 1500),
            chunk_overlap=kwargs.get('chunk_overlap', 100)
        )
    elif strategy == 'by_sentence':
        from langchain.text_splitter import SentenceSplitter
        return SentenceSplitter()

if __name__ == '__main__':
    t_all = time.time()

    data_path = 'Data/Health/'
    encoder_model_path = "H:/Models/bge-base-en-v1.5"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. 遍历文件
    t0 = time.time()
    all_files = list(find_all_file(data_path))
    print(f"共发现 {len(all_files)} 个文件，正在读取并切分……")
    documents = []
    for file_name in tqdm(all_files, desc="读取文件"):
        encoding = get_encoding_of_file(file_name)
        loader = TextLoader(file_name, encoding=encoding)
        doc = loader.load()
        documents.extend(doc)
    print(f"读取文件耗时: {time.time() - t0:.2f} 秒")

    # 2. 切分文档
    t1 = time.time()
    split_strategy = 'linebreak'
    splitter = get_text_splitter(split_strategy, chunk_size=4000, chunk_overlap=300)
    print(f"共需切分 {len(documents)} 条文档……")
    split_texts = []
    for doc in tqdm(documents, desc="切分文档"):
        split_texts.extend(splitter.split_documents([doc]))
    print(f"切分耗时: {time.time() - t1:.2f} 秒")
    print(f"共获得 {len(split_texts)} 个chunk。")

    # 3. 向量化与写入Chroma库
    t2 = time.time()
    embed_model = HuggingFaceEmbeddings(
        model_name=encoder_model_path,
        model_kwargs={'device': device},
        encode_kwargs={'device': device, 'batch_size': 256}
    )
    vector_store_path = "./RetrievalBase/Health/BASE"
    os.makedirs(vector_store_path, exist_ok=True)

    print(f"开始写入向量数据库（共{len(split_texts)}个chunk）……")
    db = Chroma.from_documents(
        documents=split_texts,
        embedding=embed_model,
        persist_directory=vector_store_path,
        collection_name="Health"
    )
    print(f"向量化与写入耗时: {time.time() - t2:.2f} 秒")
    print("chatdoctor 向量数据库已构建并保存！")
    print(f"全流程总耗时: {time.time() - t_all:.2f} 秒")
