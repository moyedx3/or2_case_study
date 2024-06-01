import multiprocessing
from easygoogletranslate import EasyGoogleTranslate
from functools import partial
from tqdm import tqdm
import requests
import json
import os
import sys
import urllib.request
client_id = "pyq4fq6nd8"
client_secret = "xmySeb2gSKckt6v4ERpUIKw51KqJDQ3xiMBQv1Em"
def translate_title(data, translator):

    try:
        encText = urllib.parse.quote(data['title'])
        trans_data = "source=ko&target=en&text=" + encText
        url = "https://naveropenapi.apigw.ntruss.com/nmt/v1/translation"
        request = urllib.request.Request(url)
        request.add_header("X-NCP-APIGW-API-KEY-ID",client_id)
        request.add_header("X-NCP-APIGW-API-KEY",client_secret)
        response = urllib.request.urlopen(request, data=trans_data.encode("utf-8"))
        rescode = response.getcode()
        if(rescode==200):
            response_body = response.read()
            dictionary = json.loads(response_body.decode('utf-8'))
            translatedText = dictionary['message']['result']['translatedText']
        else:
            print("Error Code:" + rescode)
        data['english_title'] = translatedText
    except:
        data['english_title'] = 'Not translate'

    return data

def worker(data, translator):
    return translate_title(data, translator)

def main(data_list, num_workers):
    translator = EasyGoogleTranslate(
        source_language='ko',
        target_language='en',
        timeout=10
    )

    with multiprocessing.Pool(processes=num_workers) as pool:
        worker_with_translator = partial(worker, translator=translator)
        
        results = []
        for result in tqdm(pool.imap(worker_with_translator, data_list), total=len(data_list), desc="Translating"):
            results.append(result)

    with open('output_trans.jsonl', 'w', encoding='utf-8') as f:
        for data in results:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    for data in results:
        print(f"Original: {data['title']}")
        print(f"Translated: {data['english_title']}\n")

if __name__ == '__main__':
    num_workers = 32
    file_path = '/home/minwook0008/please/OR2/or2_case_study/finbert_finetuning/samling_news_file.jsonl'

    all_data = []
    with open(file_path, 'r') as file:
        for line in file:
            all_data.append(json.loads(line))

    all_data = all_data

    main(all_data, num_workers)
