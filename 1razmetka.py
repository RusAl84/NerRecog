# Файл готовит из файла с данными  - разметку для файла
# из gdata_???.cvs делает ner_my (senteses)
# gdata_10000 сконвертировал в ner_my и разметил вручную
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
    columns = ['safeguards_txt', 'pd_category', 'pd_handle',
               'category_sub_txt', 'actions_category', 'stop_condition']
    # df = pd.read_csv("gdata_10000.csv", encoding='utf-8')
    df = pd.read_csv("gdata_edu.csv", encoding='utf-8')
    text = df.loc[:, ["safeguards_txt"]]
    text = text.values.tolist()
    stext = text
    segmenter = Segmenter()
    emb = NewsEmbedding()
    morph_tagger = NewsMorphTagger(emb)
    ind = 1
    mas_sentenses_num = []
    mas_text = []
    mas_pos = []

    for item_text in stext:
        # print(item_text[0])
        # text = text[3][0]

        text = str(item_text[0])
        # print(text)
        print(ind)
        # ind+=1
        sent = list(sentenize(text))
        for item in sent:
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
            ind += 1
    sdf = pd.DataFrame(mas_sentenses_num, columns=['mas_sentenses_num'])
    sdf['mas_text'] = pd.Series(mas_text)
    sdf['mas_pos'] = pd.Series(mas_pos)
    sdf.to_csv("ner_my.csv")
    # str1=""
    # for item in mas_text:
    #     str1 = str1 +" " + item
    # print("============================")
    # print(str1)
