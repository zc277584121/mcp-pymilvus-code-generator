# uv run process_docs_to_milvus.py --docs-dir /Users/zilliz/Downloads/web-content-master/v2.5.x/site/en/userGuide --milvus-uri http://10.100.30.11:19530 --collection pymilvus_user_guide --output-csv userGuide_embeddings.csv
# uv run process_docs_to_milvus.py --docs-dir /Users/zilliz/Downloads/web-content-master/API_Reference/pymilvus/v2.5.x/ORM --milvus-uri http://10.100.30.11:19530 --collection mcp_orm --output-csv ORM_embeddings.csv
# uv run process_docs_to_milvus.py --docs-dir /Users/zilliz/Downloads/web-content-master/API_Reference/pymilvus/v2.5.x/MilvusClient --milvus-uri http://10.100.30.11:19530 --collection mcp_milvus_client --output-csv MilvusClient_embeddings.csv


import os
import argparse
from md_2_embedding import generate_embeddings
from insert_embedding_2_vector_db import (
    create_milvus_client,
    read_embeddings_csv,
    create_collection,
    create_index,
    insert_data
)

def process_docs_to_milvus(
    docs_dir_path,
    milvus_uri,
    milvus_token,
    collection_name,
    output_csv="embeddings.csv"
):
    """
    Process markdown documents to embeddings and insert them into Milvus.
    
    Args:
        docs_dir_path: Path to the directory containing markdown files
        milvus_uri: URI of the Milvus server
        collection_name: Name of the collection to create/use in Milvus
        output_csv: Name of the CSV file to store embeddings (default: embeddings.csv)
    """
    print(f"Step 1: Converting documents to embeddings...")
    # Generate embeddings and save to CSV
    generate_embeddings(docs_dir_path, output_csv)
    
    print(f"\nStep 2: Inserting embeddings into Milvus...")
    # Create Milvus client
    client = create_milvus_client(milvus_uri, milvus_token)
    
    # Read the embeddings CSV
    df, max_content_length = read_embeddings_csv(output_csv)
    
    # Get embedding dimension from first row
    sample_embedding = eval(df.iloc[0]["embedding"])
    dim = len(sample_embedding)
    print(f"Embedding dimension: {dim}")
    
    # Create collection and indexes
    create_collection(client, collection_name, max_content_length, dim)
    create_index(client, collection_name)
    
    # Insert data
    insert_data(client, collection_name, df)
    
    # Load collection and get stats
    client.load_collection(collection_name)
    stats = client.get_collection_stats(collection_name)
    print("\nCollection statistics:")
    print(stats)
    
    print("\nProcess completed successfully!")

def main():
    parser = argparse.ArgumentParser(
        description='处理Markdown文档：生成嵌入向量并存入Milvus数据库'
    )
    
    parser.add_argument(
        '--docs-dir',
        '-d',
        required=True,
        help='Markdown文档所在的目录路径'
    )
    
    parser.add_argument(
        '--milvus-uri',
        default=None,
        help='Milvus服务器地址 (默认: http://localhost:19530)',
    )
    parser.add_argument(
        '--milvus-token',
        default=None,
        help="Milvus服务器token"
    )
    
    parser.add_argument(
        '--collection',
        '-c',
        required=True,
        help='Milvus集合名称'
    )
    
    parser.add_argument(
        '--output-csv',
        '-o',
        default='embeddings.csv',
        help='中间CSV文件的保存路径 (默认: embeddings.csv)'
    )
    
    args = parser.parse_args()
    
    # 检查文档目录是否存在
    if not os.path.exists(args.docs_dir):
        print(f"错误：文档目录不存在: {args.docs_dir}")
        return
    
    # Check and set Milvus URI from args or environment variables
    if not args.milvus_uri:
        args.milvus_uri = os.getenv('MILVUS_ENDPOINT') or os.getenv('ZILLIZ_CLOUD_URI')
        if not args.milvus_uri:
            print("Error: No Milvus URI provided. Please specify --milvus-uri or set MILVUS_ENDPOINT/ZILLIZ_CLOUD_URI environment variable")
            return
    # Check and set Milvus token from args or environment variables
    if not args.milvus_token:
        args.milvus_token = os.getenv('MILVUS_TOKEN') or os.getenv('ZILLIZ_CLOUD_API_KEY')
        if not args.milvus_token:
            print("Error: No Milvus token provided. Please specify --milvus-token or set MILVUS_TOKEN/ZILLIZ_CLOUD_API_KEY environment variable")
            return
    # 执行处理流程
    process_docs_to_milvus(
        docs_dir_path=args.docs_dir,
        milvus_uri=args.milvus_uri,
        milvus_token=args.milvus_token,
        collection_name=args.collection,
        output_csv=args.output_csv
    )

if __name__ == "__main__":
    main() 