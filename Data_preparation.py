import argparse
import pandas as pd
import os
from pathlib import Path
import nltk
import re
import argparse
import random


parser = argparse.ArgumentParser()  # Относительно удобный парсер для запуска из терминала
arg = parser.add_argument  # Принимает два аргумента, это пути к папкам,
# в которых лежат txt файлы с произведениями гоголя и гегеля
arg('--gogol_path', default="./raw_data/Nikolai Vasil'ievich Goghol'", type=str)
arg('--gegel_path', default="./raw_data/gegel/", type=str)
args = parser.parse_args()


# GOGOL_PATH = "./raw_data/Nikolai Vasil'ievich Goghol'"
# GEGEL_PATH = "./raw_data/gegel/"
GOGOL_PATH = args.gogol_path
GEGEL_PATH = args.gegel_path


def clean(string):
    """Функция для ощищения тестов от различных артефактов и мусора,
     возникшие вследствии донвертации из pdf и djvu в txt"""

    reader_artefact1 = "\xad"
    reader_artefact2 = "\n"
    reader_artefact3 = "<"
    reader_artefact4 = ">"
    reader_artefact5 = "{"
    reader_artefact6 = "}"
    reader_artefact7 = "["
    reader_artefact8 = "]"
    string = re.sub(f'{reader_artefact1}|{reader_artefact3}|{reader_artefact4}|'
                    f'{reader_artefact5}|{reader_artefact6}|{reader_artefact7}|'
                    f'{reader_artefact8}', ' ', string)

    return re.sub(f'{reader_artefact2}', ' . ', string)


def text_finder(path: Path, seq: list):
    """Рекурсивная функция, бродит по всем папкам из заданого пути, считывая и очищая все найденные txt"""

    tokenizer = nltk.data.load('https://raw.githubusercontent.com/mhq/train_punkt/master/russian.pickle')
    # Дабы не изобретать свой велосипед, скачаем чужой проверенный)

    for pth in path.iterdir():
        
        if pth.is_dir():
            text_finder(pth, seq)  # рекурсивно ходим по папкам, если находим txt добавляем в датасет
            
        elif str(pth)[-4:] == ".txt":
            text = open(str(pth), mode="r+")
            sentences = tokenizer.tokenize(clean(text.read()))
            
            for sen in sentences[len(sentences) // 20: -len(sentences) // 40]:
                # Выкидываем первые 5% и последние 2.5% текста, тк там служебная инфа, огловление и всякое такое
                if len(sen) > 10:
                    seq.append(sen)
            
    return seq


print("Processing starts, wait ~15 seconds")

# Отдельные предложения с привязкой к автору сохраняем в массивы
gogol_path = Path(GOGOL_PATH)
gogol_seq = []
text_finder(gogol_path, gogol_seq)

gegel_path = Path(GEGEL_PATH)
gegel_seq = []
text_finder(gegel_path, gegel_seq)

# Делаем upsampling
if len(gogol_seq) > len(gegel_seq):
    gegel_seq += random.sample(gegel_seq, len(gogol_seq) - len(gegel_seq))
else:
    gogol_seq += random.sample(gogol_seq, len(gegel_seq) - len(gogol_seq))


# Создаем привычную csv-шку
df = pd.DataFrame()
df["text"] = gogol_seq + gegel_seq

df.loc[:len(gogol_seq), "is_gogol"] = 1
df.loc[len(gogol_seq):, "is_gogol"] = 0

# Сохраняем в папку дата
try:
    os.mkdir("data")
except FileExistsError:
    df.to_csv(path_or_buf="./data/New_Data_Frame.csv", index=False)
else:
    df.to_csv(path_or_buf="./data/New_Data_Frame.csv", index=False)
