from langchain_text_splitters import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings


text = """LangChain was launched in October 2022 as an open source project by Harrison Chase, while working at machine learning startup Robust Intelligence. In April 2023, LangChain had incorporated and the new startup raised over $20 million in funding at a valuation of at least $200 million from venture firm Sequoia Capital, a week after announcing a $10 million seed investment from Benchmark.[3][4]

In the third quarter of 2023, the LangChain Expression Language (LCEL) was introduced, which provides a declarative way to define chains of actions.[5][6]

In October 2023 LangChain introduced LangServe, a deployment tool to host LCEL code as a production-ready API.[7]

In February 2024 LangChain released LangSmith, a closed-source observability and evaluation platform for LLM applications, and announced a US $25 million Series A led by Sequoia Capital.[8] On 14 May 2025 the company launched LangGraph Platform into general availability, providing managed infrastructure for deploying long-running, stateful AI agents.[9]

In April 2025, LangChain was featured in the Forbes AI 50 list.
The Kite Runner is the debut novel of Afghan-American author Khaled Hosseini.[3] Published in 2003 by Riverhead Books, it tells the story of Amir, a young Afghan boy from Wazir Akbar Khan, Kabul. The story is set against a backdrop of tumultuous events, beginning with the collapse of Afghanistan's monarchy and the Afghan conflict that sparked shortly thereafter, with a particular focus on the Soviet–Afghan War and the exodus of Afghan refugees, as well as the rise of the Taliban regime.
"""

# Fixed Chunking
# ----------------------------

fixed = CharacterTextSplitter(
    separator = "",
    chunk_size= 100,
    chunk_overlap=0
)

print("\n---------Fixed Size Chunking with 100 characters---------")
for i, chunk in enumerate(fixed.split_text(text), 1):
    print(f"\nChunk {i} :- ")
    print(chunk)
    print("-------------------------")


# Paragraph Chunking
# ----------------------------

paragraph = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=100,
    chunk_overlap=0
)

print("\n---------Paragraph Chunking with 100 as chunk size---------")
for i,chunk in enumerate(paragraph.split_text(text), 1):
    print(f"\nChunk {i} :- ")
    print(chunk)
    print("-------------------------")

# Recursive Chunking
# ----------------------------

recursive = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap = 20
)

print("\n---------Recursive Chunking with 100 as chunk size---------")
for i,chunk in enumerate(recursive.split_text(text), 1):
    print(f"\nChunk {i} :- ")
    print(chunk)
    print("-------------------------")


    # Semantic Chunking
# ----------------------------

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

semantic = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",  # other options: "standard_deviation", "interquartile", "gradient"
    breakpoint_threshold_amount=70
)

print("\n---------Semantic Chunking---------")
for i, chunk in enumerate(semantic.split_text(text), 1):
    print(f"\nChunk {i} :- ")
    print(chunk)
    print("-------------------------")