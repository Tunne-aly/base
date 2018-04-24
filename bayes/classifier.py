import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
import sklearn
import nltk

import csv
import argparse

import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Input file or folder")
    parser.add_argument("-f", "--folder", action="store_true", help="Data is from entire folder")
    parser.add_argument("-b", "--balance", help="Balance data for bad/good reviews")
    parser.add_argument("-d", "--data", help="Column of csv for the data")
    parser.add_argument("-k", "--key", help="Column of csv for the key")
    parser.add_argument("-o", "--output", help="Output filename")
    args = parser.parse_args()
    file_path = args.file
    reviews = pd.Series([])
    if(args.folder):
        for filename in os.listdir(file_path):
            reviews = pd.concat([reviews, read_csv(file_path + filename)])
    else:
        reviews = read_csv(file_path)

    print(reviews.shape)
    reviews = reviews[reviews['key'] != 3]
    reviews['key'] = np.where(reviews['key']<3,0,1)
    positive = reviews[reviews['key'] == 1]
    negative = reviews[reviews['key'] == 0]

    maximum = min(len(positive), len(negative))
    print("Max: " + str(maximum))

    
    
    #np.random.shuffle(negative)
    positive = positive.sample(n=maximum)
    negative = negative.sample(n=maximum)

    reviews = pd.concat([positive, negative])

    counts = reviews['key'].value_counts()
    print(counts)
    print(counts[0]/len(reviews))
    print(counts[1]/len(reviews))


    stop_words_fi=["aiemmin","aika","aikaa","aikaan","aikaisemmin","aikaisin","aikajen","aikana","aikoina","aikoo","aikovat","aina","ainakaan","ainakin","ainoa","ainoat","aiomme","aion","aiotte","aist","aivan","ajan","alas","alemmas","alkuisin","alkuun","alla","alle","aloitamme","aloitan","aloitat","aloitatte","aloitattivat","aloitettava","aloitettevaksi","aloitettu","aloitimme","aloitin","aloitit","aloititte","aloittaa","aloittamatta","aloitti","aloittivat","alta","aluksi","alussa","alusta","annettavaksi","annetteva","annettu","ansiosta","antaa","antamatta","antoi","aoua","apu","asia","asiaa","asian","asiasta","asiat","asioiden","asioihin","asioita","asti","avuksi","avulla","avun","avutta","edelle","edelleen","edellä","edeltä","edemmäs","edes","edessä","edestä","ehkä","ei","eikä","eilen","eivät","eli","ellei","elleivät","ellemme","ellen","ellet","ellette","emme","en","enemmän","eniten","ennen","ensi","ensimmäinen","ensimmäiseksi","ensimmäisen","ensimmäisenä","ensimmäiset","ensimmäisiksi","ensimmäisinä","ensimmäisiä","ensimmäistä","ensin","entinen","entisen","entisiä","entisten","entistä","enää","eri","erittäin","erityisesti","eräiden","eräs","eräät","esi","esiin","esillä","esimerkiksi","et","eteen","etenkin","etessa","ette","ettei","että","haikki","halua","haluaa","haluamatta","haluamme","haluan","haluat","haluatte","haluavat","halunnut","halusi","halusimme","halusin","halusit","halusitte","halusivat","halutessa","haluton","he","hei","heidän","heidät","heihin","heille","heillä","heiltä","heissä","heistä","heitä","helposti","heti","hetkellä","hieman","hitaasti","hoikein","huolimatta","huomenna","hyvien","hyviin","hyviksi","hyville","hyviltä","hyvin","hyvinä","hyvissä","hyvistä","hyviä","hyvä","hyvät","hyvää","hän","häneen","hänelle","hänellä","häneltä","hänen","hänessä","hänestä","hänet","häntä","ihan","ilman","ilmeisesti","itse","itsensä","itseään","ja","jo","johon","joiden","joihin","joiksi","joilla","joille","joilta","joina","joissa","joista","joita","joka","jokainen","jokin","joko","joksi","joku","jolla","jolle","jolloin","jolta","jompikumpi","jona","jonka","jonkin","jonne","joo","jopa","jos","joskus","jossa","josta","jota","jotain","joten","jotenkin","jotenkuten","jotka","jotta","jouduimme","jouduin","jouduit","jouduitte","joudumme","joudun","joudutte","joukkoon","joukossa","joukosta","joutua","joutui","joutuivat","joutumaan","joutuu","joutuvat","juuri","jälkeen","jälleen","jää","kahdeksan","kahdeksannen","kahdella","kahdelle","kahdelta","kahden","kahdessa","kahdesta","kahta","kahteen","kai","kaiken","kaikille","kaikilta","kaikkea","kaikki","kaikkia","kaikkiaan","kaikkialla","kaikkialle","kaikkialta","kaikkien","kaikkin","kaksi","kannalta","kannattaa","kanssa","kanssaan","kanssamme","kanssani","kanssanne","kanssasi","kauan","kauemmas","kaukana","kautta","kehen","keiden","keihin","keiksi","keille","keillä","keiltä","keinä","keissä","keistä","keitten","keittä","keitä","keneen","keneksi","kenelle","kenellä","keneltä","kenen","kenenä","kenessä","kenestä","kenet","kenettä","kennessästä","kenties","kerran","kerta","kertaa","keskellä","kesken","keskimäärin","ketkä","ketä","kiitos","kohti","koko","kokonaan","kolmas","kolme","kolmen","kolmesti","koska","koskaan","kovin","kuin","kuinka","kuinkan","kuitenkaan","kuitenkin","kuka","kukaan","kukin","kukka","kumpainen","kumpainenkaan","kumpi","kumpikaan","kumpikin","kun","kuten","kuuden","kuusi","kuutta","kylliksi","kyllä","kymmenen","kyse","liian","liki","lisäksi","lisää","lla","luo","luona","lähekkäin","lähelle","lähellä","läheltä","lähemmäs","lähes","lähinnä","lähtien","läpi","mahdollisimman","mahdollista","me","meidän","meidät","meihin","meille","meillä","meiltä","meissä","meistä","meitä","melkein","melko","menee","meneet","menemme","menen","menet","menette","menevät","meni","menimme","menin","menit","menivät","mennessä","mennyt","menossa","mihin","mikin","miksi","mikä","mikäli","mikään","mille","milloin","milloinkan","millä","miltä","minkä","minne","minua","minulla","minulle","minulta","minun","minussa","minusta","minut","minuun","minä","missä","mistä","miten","mitkä","mitä","mitään","moi","molemmat","mones","monesti","monet","moni","moniaalla","moniaalle","moniaalta","monta","muassa","muiden","muita","muka","mukaan","mukaansa","mukana","mutta","muu","muualla","muualle","muualta","muuanne","muulloin","muun","muut","muuta","muutama","muutaman","muuten","myöhemmin","myös","myöskin","myöskään","myötä","ne","neljä","neljän","neljää","niiden","niihin","niiksi","niille","niillä","niiltä","niin","niinä","niissä","niistä","niitä","noiden","noihin","noiksi","noilla","noille","noilta","noin","noina","noissa","noista","noita","nopeammin","nopeasti","nopeiten","nro","nuo","nyt","näiden","näihin","näiksi","näille","näillä","näiltä","näin","näinä","näissä","näissähin","näissälle","näissältä","näissästä","näistä","näitä","nämä","ohi","oikea","oikealla","oikein","ole","olemme","olen","olet","olette","oleva","olevan","olevat","oli","olimme","olin","olisi","olisimme","olisin","olisit","olisitte","olisivat","olit","olitte","olivat","olla","olleet","olli","ollut","oma","omaa","omaan","omaksi","omalle","omalta","oman","omassa","omat","omia","omien","omiin","omiksi","omille","omilta","omissa","omista","on","onkin","onko","ovat","paikoittain","paitsi","pakosti","paljon","paremmin","parempi","parhaillaan","parhaiten","perusteella","peräti","pian","pieneen","pieneksi","pienelle","pienellä","pieneltä","pienempi","pienestä","pieni","pienin","poikki","puolesta","puolestaan","päälle","runsaasti","saakka","sadam","sama","samaa","samaan","samalla","samallalta","samallassa","samallasta","saman","samat","samoin","sata","sataa","satojen","se","seitsemän","sekä","sen","seuraavat","siellä","sieltä","siihen","siinä","siis","siitä","sijaan","siksi","sille","silloin","sillä","silti","siltä","sinne","sinua","sinulla","sinulle","sinulta","sinun","sinussa","sinusta","sinut","sinuun","sinä","sisäkkäin","sisällä","siten","sitten","sitä","ssa","sta","suoraan","suuntaan","suuren","suuret","suuri","suuria","suurin","suurten","taa","taas","taemmas","tahansa","tai","takaa","takaisin","takana","takia","tallä","tapauksessa","tarpeeksi","tavalla","tavoitteena","te","teidän","teidät","teihin","teille","teillä","teiltä","teissä","teistä","teitä","tietysti","todella","toinen","toisaalla","toisaalle","toisaalta","toiseen","toiseksi","toisella","toiselle","toiselta","toisemme","toisen","toisensa","toisessa","toisesta","toista","toistaiseksi","toki","tosin","tuhannen","tuhat","tule","tulee","tulemme","tulen","tulet","tulette","tulevat","tulimme","tulin","tulisi","tulisimme","tulisin","tulisit","tulisitte","tulisivat","tulit","tulitte","tulivat","tulla","tulleet","tullut","tuntuu","tuo","tuohon","tuoksi","tuolla","tuolle","tuolloin","tuolta","tuon","tuona","tuonne","tuossa","tuosta","tuota","tuotä","tuskin","tykö","tähän","täksi","tälle","tällä","tällöin","tältä","tämä","tämän","tänne","tänä","tänään","tässä","tästä","täten","tätä","täysin","täytyvät","täytyy","täällä","täältä","ulkopuolella","usea","useasti","useimmiten","usein","useita","uudeksi","uudelleen","uuden","uudet","uusi","uusia","uusien","uusinta","uuteen","uutta","vaan","vahemmän","vai","vaiheessa","vaikea","vaikean","vaikeat","vaikeilla","vaikeille","vaikeilta","vaikeissa","vaikeista","vaikka","vain","varmasti","varsin","varsinkin","varten","vasen","vasenmalla","vasta","vastaan","vastakkain","vastan","verran","vielä","vierekkäin","vieressä","vieri","viiden","viime","viimeinen","viimeisen","viimeksi","viisi","voi","voidaan","voimme","voin","voisi","voit","voitte","voivat","vuoden","vuoksi","vuosi","vuosien","vuosina","vuotta","vähemmän","vähintään","vähiten","vähän","välillä","yhdeksän","yhden","yhdessä","yhteen","yhteensä","yhteydessä","yhteyteen","yhtä","yhtäälle","yhtäällä","yhtäältä","yhtään","yhä","yksi","yksin","yksittäin","yleensä","ylemmäs","yli","ylös","ympäri","älköön","älä"]
    review_vec = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize, stop_words=frozenset(stop_words_fi))
    review_counts = review_vec.fit_transform(reviews.data)

    tfidf_transformer = TfidfTransformer()
    review_tfidf = tfidf_transformer.fit_transform(review_counts)

    print(review_tfidf.shape)


    X_train, X_test, y_train, y_test = train_test_split(review_tfidf, reviews.key, test_size=0.2)

    X_train = X_train.todense()
    X_test = X_test.todense()
    print("Gaussian")
    gauss = GaussianNB()

    gauss.fit(
        X_train,
        y_train
    )


   
    pred = gauss.predict(X_test)
    print(sklearn.metrics.accuracy_score(y_test, pred))


    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, pred)
    print(cm)

    print("Multinomial")
    multi = MultinomialNB()
    multi.fit(
        X_train,
        y_train
    )

    pred = multi.predict(X_test)
    print(sklearn.metrics.accuracy_score(y_test, pred))
    cm = confusion_matrix(y_test, pred)
    print(cm)

    print("Bernoulli")
    bern = BernoulliNB()
    bern.fit(
        X_train,
        y_train
    )

    pred = bern.predict(X_test)
    print(sklearn.metrics.accuracy_score(y_test, pred))
    cm = confusion_matrix(y_test, pred)
    print(cm)

def read_csv(file):
    print("Loading file: " + file)
    return pd.read_csv(file, header=None, usecols=[0,1], names=["key", "data"])
def output_file(data, file):
    return 0

if __name__ == "__main__":
    main()


