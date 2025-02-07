import requests

EAS_URL = 'http://5540031936077009.ap-southeast-1.pai-eas.aliyuncs.com/api/predict/deepseek_r1_distill_qwen_14b'
headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json',
    'Authorization': 'NTFkZDMzNGE1M2FmOGJmMzllNzU1MDZjYWJjMzc5NWNlYWQwNzBlNw==',
}


def test_post_api_query_llm():
    url = EAS_URL + '/service/query/llm'
    data = {
       "question":"What is PAI?"
    }
    response = requests.post(url, headers=headers, json=data)

    if response.status_code != 200:
        raise ValueError(f'Error post to {url}, code: {response.status_code}')
    ans = dict(response.json())
    print(f"======= Question =======\n {data['question']}")
    print(f"======= Answer =======\n {ans['answer']} \n\n")


def test_post_api_query_retrieval():
    url = EAS_URL + '/service/query/retrieval'
    data = {
       "question":"What is PAI?"
    }
    response = requests.post(url, headers=headers, json=data)

    if response.status_code != 200:
        raise ValueError(f'Error post to {url}, code: {response.status_code}')
    ans = dict(response.json())
    print(f"======= Question =======\n {data['question']}")
    print(f"======= Answer =======\n {ans['docs']}\n\n")


def test_post_api_query_rag():
    url = EAS_URL + '/service/query'
    data = {
       "question":"What is PAI?"
    }
    response = requests.post(url, headers=headers, json=data)

    if response.status_code != 200:
        raise ValueError(f'Error post to {url}, code: {response.status_code}')
    ans = dict(response.json())
    print(f"======= Question =======\n {data['question']}")
    print(f"======= Answer =======\n {ans['answer']}")
    print(f"======= Retrieved Docs =======\n {ans['docs']}\n\n")
# LLM
test_post_api_query_llm()
# Retrieval
test_post_api_query_retrieval()
# RAG (Retrieval + LLM)
test_post_api_query_rag()