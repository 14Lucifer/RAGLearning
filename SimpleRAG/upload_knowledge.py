import os
import yaml
import psycopg2
from sentence_transformers import SentenceTransformer

# Load configuration from YAML file
with open("config.yaml", "r") as file:
    config                      = yaml.safe_load(file)
    dbname                      = config['pgvector']['dbname']
    user                        = config['pgvector']['user']
    password                    = config['pgvector']['password']
    host                        = config['pgvector']['host']
    port                        = config['pgvector']['port']
    sentence_transformer_model  = config['sentence_transformer_model']

# pgvector connection
def get_postgres_connection():
    return psycopg2.connect(
        dbname      =dbname,
        user        =user,
        password    =password,
        host        =host,
        port        =port
    )

# Load embedding model
embedding_model = SentenceTransformer(sentence_transformer_model)

# Folder containing text files
KNOWLEDGE_FOLDER = "knowledge_data"

def store_document_in_db(content, embedding):
    # Store a document and its embedding in PostgreSQL.
    conn = get_postgres_connection()
    cur = conn.cursor()
    
    cur.execute(
        "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
        (content, embedding)
    )
    
    conn.commit()
    cur.close()
    conn.close()

def process_files():
    # Read text files, embed content line by line, and store in PostgreSQL.
    if not os.path.exists(KNOWLEDGE_FOLDER):
        print(f"Folder '{KNOWLEDGE_FOLDER}' not found. Create it and add text files.")
        return

    line_count = 0

    for filename in os.listdir(KNOWLEDGE_FOLDER):
        if filename.endswith(".txt"):
            filepath = os.path.join(KNOWLEDGE_FOLDER, filename)
            with open(filepath, "r", encoding="utf-8") as file:
                lines = file.readlines()

            for line in lines:
                line = line.strip()  # Clean the line (remove leading/trailing whitespaces)
                if line:  # Skip empty lines
                    # Generate embedding for each line
                    embedding = embedding_model.encode(line).tolist()

                    # Store in the database (inserting each line as a separate record)
                    store_document_in_db(line, embedding)
                    line_count += 1

                    print(f"✅ Embedding length / dimension : {len(embedding)}. Uploaded line: {line[:30]}...")  # Print first 30 characters of line as reference

    print(f"\033[1mTotal uploaded lines : \033[0m{line_count}")

if __name__ == "__main__":
    process_files()
