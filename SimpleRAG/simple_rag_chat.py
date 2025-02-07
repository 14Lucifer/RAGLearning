import time
import yaml
import torch
import psycopg2
import numpy as np
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, CrossEncoder
from system_info import get_system_info, display_gpu_info


def retrieve_documents(user_query, top_k, top_n, conn, embedding_model, reranker):
    cur = conn.cursor()
    
    # Convert query into an embedding
    query_embedding = embedding_model.encode(user_query).tolist()
    query_embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
    
    # Retrieve documents from pgvector
    cur.execute(f"""
        SELECT content FROM documents 
        ORDER BY embedding <-> '{query_embedding_str}'::vector 
        LIMIT {top_k};
    """)

    results = cur.fetchall()
    cur.close()
    conn.close()
    
    retrieved_docs = [row[0] for row in results]

    if not retrieved_docs:
        return None  # No relevant documents found

    
    scores = reranker.predict([(user_query, doc) for doc in retrieved_docs])            # Re-Rank retrieved documents
    ranked_docs = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True) # Sort documents based on their re-rank scores
    top_documents = [doc for doc, score in ranked_docs[:top_n]]                         # Select top N documents with the highest scores

    print("\n\033[33m\033[1m[Data retrival & selection]\033[0m")
    print(f"\033[1mRetrived docs : \033[0m{retrieved_docs}")
    print(f"\033[1mScores : \033[0m{scores}")
    print(f"\033[1mTop docs : \033[0m{top_documents}")

    return top_documents


def main():

    # Display system information
    print("\nSystem Information:")
    print("-------------------")
    system_info = get_system_info()
    for key, value in system_info.items(): 
        print(f"{key}: {value}")
    display_gpu_info()  # Print GPU info.
    print("\n")         # Line breaker.


    # Load configuration from YAML file
    with open("config.yaml", "r") as file:
        config                      = yaml.safe_load(file)
        model_path                  = config['model']['path']
        model_name                  = config['model']['name']
        dbname                      = config['pgvector']['dbname']
        user                        = config['pgvector']['user']
        password                    = config['pgvector']['password']
        host                        = config['pgvector']['host']
        port                        = config['pgvector']['port']
        doc_top_k                   = config['knowledge_retrival']['top_k']
        doc_top_n                   = config['knowledge_retrival']['top_n']
        txt_max_length              = config['text_generation']['max_length']
        txt_top_k                   = config['text_generation']['top_k']
        txt_top_p                   = config['text_generation']['top_p']
        txt_tmp                     = config['text_generation']['temperature']
        sentence_transformer_model  = config['sentence_transformer_model']
        reranker_model              = config['reranker_model']


    # Load sentence transformer and re-ranker models
    embedding_model = SentenceTransformer(sentence_transformer_model)
    reranker = CrossEncoder(reranker_model)  # Re-Ranker Model to socre retrived documents


    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)


    # Configure BitsAndBytes quantization
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    # bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    # bnb_config = BitsAndBytesConfig(load_in_16bit=True)


    # Load the Model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        # quantization_config=bnb_config,  # Apply quantization.
        torch_dtype=torch.float16,
        device_map="auto"  
    )


    # Enable DataParallel for multiple GPUs
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)


    # Start chat interface
    print(f"\nChat with \033[1m{model_name}\033[0m. Type 'exit' to end the conversation.")

    while True:

        # pgvector connection
        conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)

        user_input = input("\n\033[31m\033[1m[Question : ]\n\033[0m")
        
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # Retrieve and re-rank documents
        retrieved_doc = retrieve_documents(
            user_query      =user_input,
            top_k           =doc_top_k,
            top_n           =doc_top_n,
            conn            =conn,
            embedding_model =embedding_model,
            reranker        =reranker)

        context = retrieved_doc if retrieved_doc else "No relevant document found."

        # Prepare input with retrieved context
        final_input = f"Context: {context}\n\nQuestion: {user_input}"
        
        device = next(model.parameters()).device
        inputs = tokenizer(final_input, return_tensors="pt").to(device)

        start_time = time.time()  # Start timer

        # Perform text generation with configured parameters
        with torch.no_grad():
            outputs = model.module.generate(
                **inputs, 
                max_length  =txt_max_length,  # Controls response length
                top_k       =txt_top_k,       # Consider top 50 tokens (increase diversity)
                top_p       =txt_top_p,       # Use nucleus sampling (filter by probability mass)
                temperature =txt_tmp,         # Add randomness (higher = more creative)
                do_sample   =True             # Enables sampling instead of greedy decoding
            )

        elapsed_time = time.time() - start_time  # End time

        # Decode the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print("\n")
        print(f"\033[32m\033[1m[Model Response : ]\n\033[0m{response}")
        print(f"\033[1m* Time taken : \033[0m{elapsed_time:.4f} seconds")
        
        # Empty GPU cache
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()