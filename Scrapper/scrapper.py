import requests
from bs4 import BeautifulSoup
import pandas as pd
from pandas import ExcelWriter

def delete_command(text, command):
    while(True):
        start = text.find('<'+command[0], 0, len(text))
        if(start == -1):
            break
        end = text.find('</'+command[0]+'>', start, len(text))
        text = text.replace(text[start:end + command[1]], ' ', 1)
    return text

def get_text_from_html(text, dataset):
    commands = [('script', 9), ('span', 7), ('style', 8)]
    for i in commands:
        text = delete_command(text, i)
    soap = BeautifulSoup(text, features='lxml').get_text()
    soap = soap.replace('\n', '')
    scores = findall_score(soap, dataset)
    return scores

def findall_score(soap, dataset):
    for i in dataset:
        start = soap.find('score:', 0, len(soap))
        if(start == -1):
            break
        end = soap.find('-', start, len(soap))
        dataset[i].append((int(soap[start+7:end-1])))
        soap = soap.replace(soap[start:end], ' ', 1)
    return dataset

def scrapper(urls):
    dataset = { 
            'Neurosis': [], 'Ansiedad': [], 'Ira': [], 'Depresion': [], 'Vergüenza': [], 'Falta_de_Moderacion': [], 'Vulnerabilidad': [],
            'Extroversion': [], 'Cordialidad': [], 'Sociabilidad': [], 'Confianza': [], 'Nivel_de_Actividad': [], 'Apertura_a_nuevas_experiencias': [], 'Alegria': [],
            'Apertura_a_experiancias': [], 'Imaginacion': [], 'Interes_Artistico': [], 'Sensibilidad': [], 'Ansias_de_aventura': [], 'Intelecto': [], 'Liberalismo': [],
            'Simpatia': [], 'Confianza_simpatia': [], 'Moral': [], 'Altruismo': [], 'Cooperacion': [], 'Modestia': [], 'Empatia': [],
            'Meticulosidad': [], 'Autoeficacia': [], 'Orden': [], 'Sentido_del_deber': [], 'Orientacion_a_objetivos': [], 'Disciplina': [], 'Prudencia': []}
    for i in range(len(urls)):
        url = urls[i]
        response = requests.get(url)
        dataset = get_text_from_html(response.text, dataset)
        print(i)
    return dataset

def generate_excel(urls, rute):
    df = pd.DataFrame(scrapper(urls))
    xlsx = ExcelWriter(rute)
    df.to_excel(xlsx, 'Big Five Data', index=True)
    xlsx.close()

#Alejandra
# urls = [
#         'https://bigfive-test.com/result/6431f77e0e940e000826b0db',
#         'https://bigfive-test.com/result/58a70606a835c400c8b38e84',
#         'https://bigfive-test.com/result/6431ffaa0e940e000826b0ff',
#         'https://bigfive-test.com/result/64336f260fdce60008273ed1',
#         'https://bigfive-test.com/result/6432e128efefc00008516e69',
#         'https://bigfive-test.com/result/643373460fdce60008273eec',
#         'https://bigfive-test.com/result/643372b70fdce60008273ee7',
#         'https://bigfive-test.com/result/645b17e9f902890008673dae',
#         'https://bigfive-test.com/result/643374360fdce60008273ef7',
#         'https://bigfive-test.com/result/6432d6b97ea5290008b61884',
#         'https://bigfive-test.com/result/64489f12aae96d000880dcda',
#         'https://bigfive-test.com/result/644894888fc17a0008362da4'
#         ]

#Kevin
# urls = [
#     'https://bigfive-test.com/result/6424a37b66adcd0008402e89',
#     'https://bigfive-test.com/result/64236db66b9c700008b6acf8',
#     'https://bigfive-test.com/result/642461e5d8db9800088449ea',
#     'https://bigfive-test.com/result/6423924a43f41200087e9cad',
#     'https://bigfive-test.com/result/6424eb5bfa524d0008d3bbcf',
#     'https://bigfive-test.com/result/6424ba605008820008120208',
#     'https://bigfive-test.com/result/64278a5b2da61900099f4840',
#     'https://bigfive-test.com/result/642367da6b9c700008b6acc7',
#     'https://bigfive-test.com/result/642626e77033800008bde884',
#     'https://bigfive-test.com/result/6424b3c350088200081201cd',
#     'https://bigfive-test.com/result/642a20110cd2b20008210946',
#     'https://bigfive-test.com/result/6424776bb85f7000096208e7',
#     'https://bigfive-test.com/result/642374af6b9c700008b6ad50',
#     'https://bigfive-test.com/result/64247211b85f700009620881',
#     'https://bigfive-test.com/result/642389c343f41200087e9c4d',
#     'https://bigfive-test.com/result/642376c56b9c700008b6ad66'
# ]

#Sheyla
# urls = [
#     'https://bigfive-test.com/result/64343b230496010008ba103e',
#     'https://bigfive-test.com/result/643346b52e5b2b000885cd32',
#     'https://bigfive-test.com/result/64337af6ad1f540008772f9c',
#     'https://bigfive-test.com/result/58a70606a835c400c8b38e84',
#     'https://bigfive-test.com/result/6434d9f17c084c000804a997',
#     'https://bigfive-test.com/result/6433311841173d0008ad57a8',
#     'https://bigfive-test.com/result/6434e4617c084c000804a9d1',
#     'https://bigfive-test.com/result/643434c00496010008ba0fec',
#     'https://bigfive-test.com/result/6434ecff7c084c000804a9fb',
#     'https://bigfive-test.com/result/64337f7aad1f540008772fba',

# ]

#Andry
# urls=[
#     'https://bigfive-test.com/result/6431e0933a9f0f0008c6ef9e',
#     'https://bigfive-test.com/result/642b070a1e72b800082a7eb0',
#     'https://bigfive-test.com/result/64285dc27fba2c00087b6f48',
#     'https://bigfive-test.com/result/642dd37e086ba300097ba388',
#     'https://bigfive-test.com/result/642da34b44d5540008389d4c',
#     'https://bigfive-test.com/result/6434966152b6430008423da1',
#     'https://bigfive-test.com/result/642b8770027f4a000809e5f9',
#     'https://bigfive-test.com/result/642ddc5b086ba300097ba415',
#     'https://bigfive-test.com/result/6425f8fc29c6ed00080a263b',
#     'https://bigfive-test.com/result/6425f8d829c6ed00080a2638',
#     'https://bigfive-test.com/result/642de049ca4b7b0008313f36',
#     'https://bigfive-test.com/result/64303327b53c6e00088a2ed0',
#     'https://bigfive-test.com/result/6424d7b2c337f600084a65a3',
#     'https://bigfive-test.com/result/642db56e277d8e0008418a8a',
#     'https://bigfive-test.com/result/642601c6b6bf1a0008a91f50',
#     'https://bigfive-test.com/result/642dd2bd086ba300097ba378',
#     'https://bigfive-test.com/result/642ddef9086ba300097ba449',
#     'https://bigfive-test.com/result/642de4c3ca4b7b0008313ff4',
#     'https://bigfive-test.com/result/642dd50c086ba300097ba398',
#     'https://bigfive-test.com/result/642ddc2d086ba300097ba40f',
#     'https://bigfive-test.com/result/642ddd3c086ba300097ba41f',
#     'https://bigfive-test.com/result/642b9345027f4a000809e662',
#     'https://bigfive-test.com/result/642dd7ea086ba300097ba3c5',
#     'https://bigfive-test.com/result/642b8c9e027f4a000809e626'
# ]

#Javier
# urls=[
#     'http://bigfive-test.com/result/6431eb6d3a9f0f0008c6eff8',
#     'http://bigfive-test.com/result/643466ad56c44c00089edaa5'
# ]

#Laura
urls=[
    'https://bigfive-test.com/result/6423a18db4caeb00086c07d4',
    'https://bigfive-test.com/result/64244df8ff75de00080523bc',
    'https://bigfive-test.com/result/6424dd95c337f600084a65c0',
    'https://bigfive-test.com/result/645b8925811958000860839c',
    'https://bigfive-test.com/result/6431d5993a9f0f0008c6ef42',
    'https://bigfive-test.com/result/64244d58ff75de00080523b3',
    'https://bigfive-test.com/result/642f4a9b9dad980008a236a2'
]

rute = 'big_five_ok.xlsx'

generate_excel(urls, rute)