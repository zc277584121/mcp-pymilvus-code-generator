import logging
import os
from openai import OpenAI
from pymilvus import AnnSearchRequest, MilvusClient, WeightedRanker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("milvus-connector")


class MilvusConnector:
    def __init__(
        self,
        milvus_uri="http://localhost:19530",
        milvus_token="",
        db_name="default",
    ):
        logger.debug("Initializing MilvusConnector")
        self.milvus_client = MilvusClient(uri=milvus_uri, token=milvus_token, db_name=db_name)

        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        try:
            self.milvus_client.load_collection("pymilvus_user_guide")
            self.milvus_client.load_collection("mcp_orm")
            self.milvus_client.load_collection("mcp_milvus_client")
        except Exception as e:
            logger.error(f"Fail to load collection: {e}")

    def create_embedding(self, text):
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small", input=text
            )
            embedding = response.data[0].embedding
            return embedding
        except Exception as e:
            logger.error(f"Fail to create embedding from user query: {e}")
            return None

    def search_similar_documents(self, collection_name, query_text, top_k=10):
        query_embedding = self.create_embedding(query_text)
        if query_embedding is None:
            logger.error("Fail to create embedding from user query. Stop.")
            return []

        try:
            vector_request = AnnSearchRequest(
                data=[query_embedding],
                anns_field="dense",
                param={
                    "metric_type": "IP",
                },
                limit=top_k,
            )

            text_request = AnnSearchRequest(
                data=[query_text],
                anns_field="sparse",
                param={
                    "metric_type": "BM25",
                },
                limit=top_k,
            )

            requests = [vector_request, text_request]

            ranker = WeightedRanker(0.5, 0.5)

            results = self.milvus_client.hybrid_search(
                collection_name=collection_name,
                reqs=requests,
                ranker=ranker,
                limit=top_k,
                output_fields=["metadata", "content"],
            )

            return results

        except Exception as e:
            logger.warning(f"Hybrid search failed when searching for similar documents: {e}")
            return []

    async def pypmilvus_code_generate_helper(self, query) -> str:
        """
        Retrieve related pymilvus code/documents for a given query.

        :param query: User query for generating code in natural language
        :return: related pymilvus code/documents for generating code from user query
        """
        results = self.search_similar_documents("pymilvus_user_guide", query)

        if not results:
            logger.warning("No related document found.")
            return "No related document found."

        related_documents = "Here are related pymilvus code/documents found to help you generate code from user query:\n\n"
        for i, hit in enumerate(results[0]):
            content = hit["entity"]["content"]
            related_documents += f"{i + 1}:\n{content}\n\n"

        return related_documents

    async def orm_to_milvus_client_code_translate_helper(self, query) -> str:
        """
        Retrieve related orm and pymilvus client code/documents for a given query.

        :param query: User query for translating orm code to milvus client in natural language
        :return: related orm and pymilvus client code/documents for a user query
        """

        orm_results = self.search_similar_documents("mcp_orm", query)

        if not orm_results:
            logger.warning("No related orm document found.")
            return "No related orm document found."

        milvus_client_results = self.search_similar_documents("mcp_milvus_client", query)

        if not milvus_client_results:
            logger.warning("No related milvus client document found.")
            return "No related milvus client document found."

        related_documents = "Here are related orm and pymilvus client code/documents found to help you translate orm code to milvus client from user query:\n\n"

        related_documents += "Related ORM documents:\n\n"
        for i, hit in enumerate(orm_results[0]):
            content = hit["entity"]["content"]
            related_documents += f"{i + 1}:\n{content}\n\n"

        related_documents += "Related Milvus Client documents:\n\n"
        for i, hit in enumerate(milvus_client_results[0]):
            content = hit["entity"]["content"]
            related_documents += f"{i + 1}:\n{content}\n\n"

        return related_documents
