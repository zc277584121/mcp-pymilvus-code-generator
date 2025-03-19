from collections import defaultdict
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from server import PymilvusServer

# Given a document, generate a list query, each of which is a query string to generate code. Return in List[str] format.
QUERY_GENERATION_PROMPT1 = """
Given a document, generate a list query, each of which is a query string to generate python code. Return in List[str] format. 

Document: 

<document>
{doc}
</document>

The generated queries:
"""

QUERY_GENERATION_PROMPT2 = """
...
"""

PROMPT_LIST = [QUERY_GENERATION_PROMPT1, QUERY_GENERATION_PROMPT2]
def literal_eval(response_content: str):
    import ast
    import re

    response_content = response_content.strip()

    # remove content between <think> and </think>, especial for DeepSeek reasoning model
    if "<think>" in response_content and "</think>" in response_content:
        end_of_think = response_content.find("</think>") + len("</think>")
        response_content = response_content[end_of_think:]

    try:
        if response_content.startswith("```") and response_content.endswith("```"):
            if response_content.startswith("```python"):
                response_content = response_content[9:-3]
            elif response_content.startswith("```json"):
                response_content = response_content[7:-3]
            elif response_content.startswith("```str"):
                response_content = response_content[6:-3]
            elif response_content.startswith("```\n"):
                response_content = response_content[4:-3]
            else:
                raise ValueError("Invalid code block format")
        result = ast.literal_eval(response_content.strip())
    except Exception:
        matches = re.findall(r"(\[.*?\]|\{.*?\})", response_content, re.DOTALL)

        if len(matches) != 1:
            raise ValueError(f"Invalid JSON/List format for response content:\n{response_content}")

        json_part = matches[0]
        return ast.literal_eval(json_part)

    return result


def test_data_generation(milvus_uri="http://localhost:19530", save_path="data.json", data_root_dir=...):
    import json

    def _get_doc_by_file_name(file_name, data_root_dir):
        for root, dirs, files in os.walk(data_root_dir):
            for file in files:
                if file == file_name:
                    with open(os.path.join(root, file), "r") as f:
                        return f.read()
        assert False, f"File {file_name} not found in {data_root_dir}"
    
    def _generate_query_list_from_llm(gold_doc):
        response = server.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": QUERY_GENERATION_PROMPT.format(doc=gold_doc)},
            ],
        )
        query_list = literal_eval(response.choices[0].message.content)
        return query_list

    MAX_NUM = 16384
    server = PymilvusServer(milvus_uri=milvus_uri)
    res = server.milvus_client.query(
        collection_name="pymilvus_user_guide",
        output_fields=["content", "metadata"],
        limit=MAX_NUM,
    )

    all_docs = [row["content"] for row in res]
    all_file_name = [row["metadata"] for row in res]

    all_file_name = list(set(all_file_name))

    query_to_single_gold = {}

    for i, file_name in enumerate(all_file_name[:3]):  #TODO: remove "[:3]"
        print(f"Process {i + 1}/{len(all_file_name)} doc: {file_name}")
        gold_doc = _get_doc_by_file_name(file_name, data_root_dir)  # todo

        
        # For diversity, generate query list from different prompts and different chunked
        query_list = []
        random_chunked_gold_docs = _random_chunk_gold_docs(gold_doc)
        for chunked_gold_doc in random_chunked_gold_docs:
            for prompt in PROMPT_LIST:
                sub_query_list = _generate_query_list_from_llm(chunked_gold_doc, prompt)
                query_list.extend(sub_query_list)

                for query in query_list:
                    query_to_single_gold[query] = {
                        "file_name": file_name,
                        "gold_doc": gold_doc,
                    }
                



    # Deduplicate query list by vector embedding similarity
    queries = list(query_to_single_gold.keys())
    filtered_queries = _deduplicate_query_list(queries, similarity_threshold=...)
    query_to_single_gold = {query: query_to_single_gold[query] for query in filtered_queries}


    # For each query, find all gold docs that can answer the query
    query_to_gold_list = defaultdict(list)
    for query in query_to_single_gold.keys():
        for gold_info in query_to_single_gold.values():
            can_answer = jugde_if_gold_can_answer_query(gold_info["gold_doc"], query)
            if can_answer:
                query_to_gold_list[query].append(gold_info)
    
    # Save query_to_gold_list to json file
    data = []
    for query, gold_list in query_to_gold_list.items():
        for gold_info in gold_list:
            data.append({
                "query": query,   # Note: have changed from query_list to query
                "gold_doc": gold_info["gold_doc"],
                "file_name": gold_info["file_name"],
            })
        

    
    with open(save_path, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    test_data_generation(milvus_uri="http://10.100.30.11:19530", save_path="data.json", data_root_dir=...)
