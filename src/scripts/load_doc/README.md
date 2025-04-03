For Milvus Server, you need to set the following environment variables:
```shell
export MILVUS_ENDPOINT=your_milvus_uri
export MILVUS_TOKEN=your_milvus_token
```

For zilliz cloud, you need to set the following environment variables:
```shell
export ZILLIZ_CLOUD_URI=your_zilliz_cloud_uri
export ZILLIZ_CLOUD_API_KEY=your_zilliz_cloud_api_key
```

```shell
# cd to this folder
cd YOUR_PROJECT_PATH/src/scripts/load_doc

git clone https://github.com/milvus-io/web-content.git

uv run process_docs_to_milvus.py --docs-dir ./web-content/v2.5.x/site/en/userGuide --collection pymilvus_user_guide --output-csv userGuide_embeddings.csv
# uv run process_docs_to_milvus.py --docs-dir ./web-content/API_Reference/pymilvus/v2.5.x/ORM --collection mcp_orm --output-csv ORM_embeddings.csv
# uv run process_docs_to_milvus.py --docs-dir ./web-content/API_Reference/pymilvus/v2.5.x/MilvusClient --collection mcp_milvus_client --output-csv MilvusClient_embeddings.csv
```