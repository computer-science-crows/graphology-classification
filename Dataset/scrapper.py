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
        start = soap.find('score', 0, len(soap))
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
    return dataset

def generate_excel(urls, rute):
    df = pd.DataFrame(scrapper(urls))
    xlsx = ExcelWriter(rute)
    df.to_excel(xlsx, 'Big Five Data', index=True)
    xlsx.close()

urls = ['https://bigfive-test.com/result/64278a5b2da61900099f4840',
        'https://bigfive-test.com/result/642626e77033800008bde884',
        'https://bigfive-test.com/result/6424eb5bfa524d0008d3bbcf',
        'https://bigfive-test.com/result/6431f77e0e940e000826b0db',
        'https://bigfive-test.com/result/64336f260fdce60008273ed1',
        'https://bigfive-test.com/result/643374360fdce60008273ef7',
        'https://bigfive-test.com/result/6432d6b97ea5290008b61884',
        'https://bigfive-test.com/result/6431ffaa0e940e000826b0ff',
        'https://bigfive-test.com/result/6432e128efefc00008516e69',
        'https://bigfive-test.com/result/643372b70fdce60008273ee7',
        'https://bigfive-test.com/result/643373460fdce60008273eec',
        'https://bigfive-test.com/result/644894888fc17a0008362da4',
        'https://bigfive-test.com/result/64485252c559a20008972973',
        'https://bigfive-test.com/result/64485939c559a2000897299b',
        'https://bigfive-test.com/result/58a70606a835c400c8b38e84',
        'https://bigfive-test.com/result/64489f12aae96d000880dcda',
        'https://bigfive-test.com/result/644894888fc17a0008362da4']

rute = 'D:/UniVerSiDaD/IV Ano/Machine Learning/Project/Dataset/big_five.xlsx'

generate_excel(urls, rute)