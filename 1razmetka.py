# Файл готовит из файла с данными  - разметку для файла
# из gdata_???.cvs делает ner_my (senteses)
# gdata_10000 сконвертировал в ner_my и разметил вручную учил на 10000

# gdata_edu
import pandas as pd
from razdel import sentenize
from natasha import (
    Segmenter,
    NewsEmbedding,
    NewsMorphTagger,
    Doc
)


if __name__ == "__main__":

    # #Загрузка данных из файла csv
    # columns = ['safeguards_txt', 'pd_category', 'pd_handle',
    #            'category_sub_txt', 'actions_category', 'stop_condition']
    # df = pd.read_csv("gdata_10000.csv", encoding='utf-8')
    df = pd.read_csv("gdata_edu.csv", encoding='utf-8')
    # выбор поля с данными
    text = df.loc[:, ["safeguards_txt"]]
    text = text.values.tolist()

    # загрузка данных из файла txt
    # with open('text.txt', encoding="utf-8") as fp:
    #     data = fp.read()
    # text=[[data]]


    stext = text

    segmenter = Segmenter()
    emb = NewsEmbedding()
    morph_tagger = NewsMorphTagger(emb)
    ind = 1
    mas_sentenses_num = []
    mas_text = []
    mas_pos = []
    mas_tag = []

    for item_text in stext:
        # print(item_text[0])
        # text = text[3][0]

        text = str(item_text[0])
        # print(text)
        print(ind)
        # ind+=1
        sentens = list(sentenize(text))
        for item in sentens: # предложения
            # print("_______")
            ttext = item.text
            # print(ttext)
            doc = Doc(ttext)
            doc.segment(segmenter)
            doc.tag_morph(morph_tagger)
            # doc.sents[0].morph.print()
            sents = doc.sents[0].morph
            for item1 in sents:
                for item2 in item1:
                    mas_sentenses_num.append(str("Sentence: " + str(ind)))
                    str1 = item2.text
                    if str1==";":
                        str1 = ".,"
                    mas_text.append(str1)
                    pos = item2.pos
                    mas_pos.append(pos)
                    # print(item2.text + " " + item2.pos)
                    mas_tag.append("O")
            ind += 1
    sdf = pd.DataFrame(mas_sentenses_num, columns=['Sentence #'])
    sdf['Word'] = pd.Series(mas_text)
    sdf['POS'] = pd.Series(mas_pos)
    sdf['Tag'] = pd.Series(mas_tag)
    sdf.to_csv("ner_mydimatest.csv")


