import asyncio
import inspect
import logging
import os
import re
import time

import dotenv
import requests

from lightrag import LightRAG, QueryParam
from lightrag.llm_gemini import gemini_complete, gemini_embedding
from lightrag.utils import EmbeddingFunc

dotenv.load_dotenv()
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

WORKING_DIR = "./dickens"


def fetch_wikipedia_article(title: str) -> str:
    response = requests.get(
        "https://ja.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            "explaintext": True,
        },
    ).json()
    page_id = next(iter(response["query"]["pages"]))
    text = response["query"]["pages"][page_id]["extract"]
    text = text.replace("\n\n", "\n")
    text = re.sub(r"==\s*(.+)\s*==", r"# \1", text)
    return text


def measure_time(func_name: str, func):
    start = time.time()
    result = func()
    end = time.time()
    print(f"{func_name}の実行時間: {end - start:.4f}秒")
    return result


def invoke_sync_queries(rag: LightRAG):

    # データ登録
    text = fetch_wikipedia_article("桃太郎")
    measure_time("register data", lambda: rag.insert(text))

    # naive 検索
    measure_time(
        "naive search",
        lambda: print(rag.query("桃太郎の仲間教えて", param=QueryParam(mode="naive"))),
    )

    # local 検索
    measure_time(
        "local search",
        lambda: print(rag.query("桃太郎の仲間教えて", param=QueryParam(mode="local"))),
    )

    # # global 検索
    measure_time(
        "global search",
        lambda: print(rag.query("桃太郎の仲間教えて", param=QueryParam(mode="global"))),
    )

    # # hybrid 検索
    measure_time(
        "hybrid search",
        lambda: print(rag.query("桃太郎の仲間教えて", param=QueryParam(mode="hybrid"))),
    )


async def print_stream(stream):
    for chunk in stream:
        print(chunk, end="", flush=True)


def invoke_async_queries(rag: LightRAG):
    resp = rag.query(
        "桃太郎の仲間教えて",
        param=QueryParam(mode="hybrid", stream=True),
    )

    # キャッシュから回答された場合は、asyncgenではないので、そのままprintする
    if inspect.isasyncgen(resp):
        asyncio.run(print_stream(resp))
    else:
        print(resp)


if __name__ == "__main__":

    if not os.path.exists(WORKING_DIR):
        os.mkdir(WORKING_DIR)

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=gemini_complete,
        llm_model_name="models/gemini-2.0-flash-latest",
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=lambda texts: gemini_embedding(
                texts, embed_model="models/text-embedding-004"
            ),
        ),
    )
    invoke_sync_queries(rag)
    invoke_async_queries(rag)
