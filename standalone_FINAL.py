import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
from data import Data
import pickle
import sys

import numpy as np
import pandas as pd

'''Features'''
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as imbPipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler

'''Classifiers'''
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from tensorflow import keras
import utils as utils

import logging

'''Metrics/Evaluation'''
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, confusion_matrix
from scipy import interp
from itertools import cycle

'''Plotting'''
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
import re
import string

import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer

'''Slovene lemmtizer'''
import lemmagen.lemmatizer
from lemmagen.lemmatizer import Lemmatizer

layers = keras.layers
models = keras.models

'''Display'''
sns.set_style('darkgrid')

logging.basicConfig(filename='zurnal-last.log', format='%(asctime)s - %(message)s', level=logging.DEBUG)

global_filename = './app/labeled_urls.tsv'
dataset_path = './dataset'
model_path = './models/'

classnames_dict = {0: 'avto', 1: 'sport', 2: 'svet', 3: 'slovenija', 4: 'magazin'}

fetures_dict = [{'feature': CountVectorizer(), 'name': 'CountVectorizer'},
                ]

"""
{'feature': TfidfVectorizer(), 'name': 'TfidfVectorizer', 'level': 'word_char'},
                {'feature': TfidfVectorizer(), 'name': 'TfidfVectorizer', 'level': 'n-gram'}
"""

sampling_dict = [{'method': SMOTE(), 'name': 'SMOTE', 'params': {'k_neighbors': [1, 5, 10]}},
                 ]

"""
{'method': RandomOverSampler(), 'name': 'RandomOverSampler'},
                 {'method': RandomUnderSampler(), 'name': 'RandomUnderSampler'}
"""

reduction_dict = [
    {'method': TruncatedSVD(n_components=10, n_iter=10), 'name': 'TruncatedSVD',
     'params': {'n_iter': 20, 'random_state': 3}},
]
# {'method':NMF(), 'name': 'NMF', 'params': {'n_components':10, 'n_iter':10, 'init':'random', 'random_state':3}
"""
{'method': TruncatedSVD(n_components=50, n_iter=10), 'name': 'TruncatedSVD',
     'params': {'n_iter': 20, 'random_state': 3}},
    {'method': TruncatedSVD(n_components=100, n_iter=10), 'name': 'TruncatedSVD',
     'params': {'n_iter': 20, 'random_state': 3}},
    {'method': TruncatedSVD(n_components=500, n_iter=10), 'name': 'TruncatedSVD',
     'params': {'n_iter': 20, 'random_state': 3}}
"""

final_models = [

    {'classifier': LogisticRegression, 'name': 'LogisticRegression', 'settings': 'lr.settings',
     'filename': 'lr.mod',
     'params':
         {'C': [1.0, 0.1, 0.01], 'penalty': ['l2'],
          'max_iter': [500, 1000],
          'tol': [1e-3, 1e-4], 'solver': ['lbfgs'], 'multi_class': ['multinomial']}},
    {'classifier': SGDClassifier, 'name': 'SGDClassifier', 'settings': 'sgd.settings', 'filename': 'sgd.mod',
     'params':
         {'loss': ['log'], 'penalty': ['l2', 'l1'],
          'alpha': [1e-6, 1e-3], 'max_iter': [500, 1000],
          'tol': [None, 1e-3], 'eta0': [0.1, 0.001]}},

]
"""
    {'classifier': RandomForestClassifier, 'name': 'RandomForestClassifier', 'settings': 'rf.settings',
     'filename': 'rf.mod',
     'params':
         {'bootstrap': [True, False], 'max_depth': [10, 50, 100, None], 'max_features': ['auto', 'sqrt'],
          'min_samples_leaf': [1, 2, 4], 'min_samples_split': [2, 5, 10],
          'n_estimators': [300, 800, 1400, 2000]}}
    """

SLOVENE_STOP_WORDS = frozenset(
    ["a", "ali", "april", "avgust", "b", "bi", "bil", "bila", "bile", "bili", "bilo", "biti", "blizu", "bo", "bodo",
     "bojo", "bolj", "bom", "bomo", "boste", "bova", "boš", "brez", "c", "cel", "cela", "celi", "celo", "d", "da",
     "daleč", "dan", "danes", "datum", "december", "deset", "deseta", "deseti", "deseto", "devet", "deveta",
     "deveti", "deveto", "do", "dober", "dobra", "dobri", "dobro", "dokler", "dol", "dolg", "dolga", "dolgi", "dovolj",
     "drug", "druga", "drugi", "drugo", "dva", "dve", "e", "eden", "en", "ena", "ene", "eni", "enkrat", "eno", "etc.",
     "f", "februar", "g", "g.", "ga", "ga.", "gor", "gospa", "gospod", "h", "halo", "i", "idr.", "ii", "iii", "in",
     "iv", "ix", "iz", "j", "januar", "jaz", "je", "ji", "jih", "jim", "jo", "julij", "junij", "jutri", "k",
     "kadarkoli", "kaj", "kajti", "kako", "kakor", "kamor", "kamorkoli", "kar", "karkoli", "katerikoli", "kdaj", "kdo",
     "kdorkoli", "ker", "ki", "kje", "kjer", "kjerkoli", "ko", "koder", "koderkoli", "koga", "komu", "kot", "kratek",
     "kratka", "kratke", "kratki", "l", "lahka", "lahke", "lahki", "lahko", "le", "lep", "lepa", "lepe", "lepi", "lepo",
     "leto", "m", "maj", "majhen", "majhna", "majhni", "malce", "malo", "manj", "marec", "me", "med", "medtem", "mene",
     "mesec",  "mi", "midva", "midve", "mnogo", "moj", "moja", "moje", "mora", "morajo", "moram", "moramo", "morate",
     "moraš", "morem", "mu", "n", "na", "nad", "naj", "najina", "najino", "najmanj", "naju", "največ", "nam", "narobe",
     "nas", "nato", "nazaj", "naš", "naša", "naše", "ne", "nedavno", "nedelja", "nek", "neka", "nekaj", "nekatere",
     "nekateri", "nekatero", "nekdo", "neke", "nekega", "neki", "nekje", "neko", "nekoga", "nekoč", "ni", "nikamor",
     "nikdar", "nikjer", "nikoli", "nič", "nje", "njega", "njegov", "njegova", "njegovo", "njej", "njemu", "njen",
     "njena", "njeno", "nji", "njih", "njihov", "njihova", "njihovo", "njiju", "njim", "njo", "njun", "njuna", "njuno",
     "no", "nocoj", "november", "npr.", "o", "ob", "oba", "obe", "oboje", "od", "odprt", "odprta", "odprti", "okoli",
     "oktober", "on", "onadva", "one", "oni", "onidve", "osem", "osma", "osmi", "osmo", "oz.", "p", "pa", "pet",
     "peta", "petek", "peti", "peto", "po", "pod", "pogosto", "poleg", "poln", "polna", "polni", "polno", "ponavadi",
     "ponedeljek", "ponovno", "potem", "povsod", "pozdravljen", "pozdravljeni", "prav", "prava", "prave", "pravi",
     "pravo", "prazen", "prazna", "prazno", "prbl.", "precej", "pred", "prej", "preko", "pri", "pribl.",
     "približno", "primer", "pripravljen", "pripravljena", "pripravljeni", "proti", "prva", "prvi", "prvo", "r", "ravno",
     "redko", "res", "reč", "s", "saj", "sam", "sama", "same", "sami", "samo", "se", "sebe", "sebi", "sedaj", "sedem",
     "sedma", "sedmi", "sedmo", "sem", "september", "seveda", "si", "sicer", "skoraj", "skozi", "slab", "smo", "so",
     "sobota", "spet", "sreda", "srednja", "srednji", "sta", "ste", "stran", "stvar", "sva", "t", "ta", "tak", "taka",
     "take", "taki", "tako", "takoj", "tam", "te", "tebe", "tebi", "tega", "težak", "težka", "težki", "težko", "ti",
     "tista", "tiste", "tisti", "tisto", "tj.", "tja", "to", "toda", "torek", "tretja", "tretje", "tretji", "tri", "tu",
     "tudi", "tukaj", "tvoj", "tvoja", "tvoje", "u", "v", "vaju", "vam", "vas", "vaš", "vaša", "vaše", "ve", "vedno",
     "velik", "velika", "veliki", "veliko", "vendar", "ves", "več", "vi", "vidva", "vii", "viii", "visok", "visoka",
     "visoke", "visoki", "vsa", "vsaj", "vsak", "vsaka", "vsakdo", "vsake", "vsaki", "vsakomur", "vse", "vsega", "vsi",
     "vso", "včasih", "včeraj", "x", "z", "za", "zadaj", "zadnji", "zakaj", "zaprta", "zaprti", "zaprto", "zdaj", "zelo",
     "zunaj", "č", "če", "često", "četrta", "četrtek", "četrti", "četrto", "čez", "čigav", "š", "šest", "šesta",
     "šesti", "šesto", "štiri", "ž", "že", "ali", "ampak", "bodisi", "in", "kajti", "marveč", "namreč", "ne", "niti",
     "oziroma", "pa", "saj", "sicer", "temveč", "ter", "toda", "torej", "vendar", "vendarle", "zakaj", "če", "čeprav",
     "čeravno", "četudi", "čim", "da", "kadar", "kakor", "ker", "ki", "ko", "kot", "naj", "najsi", "odkar", "preden",
     "dve", "dvema", "dveh", "šest", "šestdeset", "šestindvajset", "šestintrideset", "šestnajst", "šeststo", "štiri", "štirideset",
     "štiriindvajset",
     "štirinajst", "štiristo", "deset", "devet", "devetdeset", "devetintrideset", "devetnajst", "devetsto",
     "dvainšestdeset", "dvaindvajset", "dvajset", "dvanajst", "dvesto", "enaindvajset", "enaintrideset", "enajst",
     "nič", "osem", "osemdeset", "oseminštirideset", "osemindevetdeset", "osemnajst", "pet", "petdeset",
     "petinštirideset", "petindevetdeset", "petindvajset", "petinosemdeset", "petinpetdeset", "petinsedemdeset",
     "petintrideset", "petnajst", "petsto", "sedem", "sedemdeset", "sedeminšestdeset", "sedemindvajset",
     "sedeminpetdeset", "sedemnajst", "sedemsto", "sto", "tisoč", "tri", "trideset", "triinšestdeset",
     "triindvajset",
     "triinpetdeset", "trinajst", "tristo", "šestdesetim", "šestim", "šestindvajsetim", "šestintridesetim",
     "šestnajstim", "šeststotim", "štiridesetim", "štiriindvajsetim", "štirim", "štirinajstim", "štiristotim",
     "desetim", "devetdesetim", "devetim", "devetintridesetim", "devetnajstim", "devetstotim", "dvainšestdesetim",
     "dvaindvajsetim", "dvajsetim", "dvanajstim", "dvestotim", "enaindvajsetim", "enaintridesetim", "enajstim",
     "osemdesetim", "oseminštiridesetim", "osemindevetdesetim", "osemnajstim", "osmim", "petdesetim", "petim",
     "petinštiridesetim", "petindevetdesetim", "petindvajsetim", "petinosemdesetim", "petinpetdesetim",
     "petinsedemdesetim", "petintridesetim", "petnajstim", "petstotim", "sedemdesetim", "sedeminšestdesetim",
     "sedemindvajsetim", "sedeminpetdesetim", "sedemnajstim", "sedemstotim", "sedmim", "stotim", "tisočim", "trem",
     "tridesetim", "triinšestdesetim", "triindvajsetim", "triinpetdesetim", "trinajstim", "tristotim",
     "šestdesetih",
     "šestih", "šestindvajsetih", "šestintridesetih", "šestnajstih", "šeststotih", "štiridesetih", "štirih",
     "štiriindvajsetih", "štirinajstih", "štiristotih", "desetih", "devetdesetih", "devetih", "devetintridesetih",
     "devetnajstih", "devetstotih", "dvainšestdesetih", "dvaindvajsetih", "dvajsetih", "dvanajstih", "dvestotih",
     "enaindvajsetih", "enaintridesetih", "enajstih", "osemdesetih", "oseminštiridesetih", "osemindevetdesetih",
     "osemnajstih", "osmih", "petdesetih", "petih", "petinštiridesetih", "petindevetdesetih", "petindvajsetih",
     "petinosemdesetih", "petinpetdesetih", "petinsedemdesetih", "petintridesetih", "petnajstih", "petstotih",
     "sedemdesetih", "sedeminšestdesetih", "sedemindvajsetih", "sedeminpetdesetih", "sedemnajstih", "sedemstotih",
     "sedmih", "stotih", "tisočih", "treh", "tridesetih", "triinšestdesetih", "triindvajsetih", "triinpetdesetih",
     "trinajstih", "tristotih", "šestdesetimi", "šestimi", "šestindvajsetimi", "šestintridesetimi", "šestnajstimi",
     "šeststotimi", "štiridesetimi", "štiriindvajsetimi", "štirimi", "štirinajstimi", "štiristotimi", "desetimi",
     "devetdesetimi", "devetimi", "devetintridesetimi", "devetnajstimi", "devetstotimi", "dvainšestdesetimi",
     "dvaindvajsetimi", "dvajsetimi", "dvanajstimi", "dvestotimi", "enaindvajsetimi", "enaintridesetimi",
     "enajstimi",
     "osemdesetimi", "oseminštiridesetimi", "osemindevetdesetimi", "osemnajstimi", "osmimi", "petdesetimi",
     "petimi",
     "petinštiridesetimi", "petindevetdesetimi", "petindvajsetimi", "petinosemdesetimi", "petinpetdesetimi",
     "petinsedemdesetimi", "petintridesetimi", "petnajstimi", "petstotimi", "sedemdesetimi", "sedeminšestdesetimi",
     "sedemindvajsetimi", "sedeminpetdesetimi", "sedemnajstimi", "sedemstotimi", "sedmimi", "stotimi", "tisočimi",
     "tremi", "tridesetimi", "triinšestdesetimi", "triindvajsetimi", "triinpetdesetimi", "trinajstimi",
     "tristotimi",
     "eno", "eni", "ene", "ena", "dva", "štirje", "trije", "en", "enega", "enemu", "enim", "enem", "eden", "dvojni",
     "trojni", "dvojnima", "trojnima", "dvojnih", "trojnih", "dvojne", "trojne", "dvojnim", "trojnim", "dvojnimi",
     "trojnimi", "dvojno", "trojno", "dvojna", "trojna", "dvojnega", "trojnega", "dvojen", "trojen", "dvojnemu",
     "trojnemu", "dvojnem", "trojnem", "četrti", "šestdeseti", "šesti", "šestnajsti", "štirideseti",
     "štiriindvajseti",
     "štirinajsti", "deseti", "devetdeseti", "deveti", "devetnajsti", "drugi", "dvaindevetdeseti", "dvajseti",
     "dvanajsti", "dvestoti", "enaindvajseti", "enajsti", "osemdeseti", "osemnajsti", "osmi", "petdeseti", "peti",
     "petinštirideseti", "petindvajseti", "petinosemdeseti", "petintrideseti", "petnajsti", "prvi", "sedemdeseti",
     "sedemindvajseti", "sedemnajsti", "sedmi", "stoti", "tisoči", "tretji", "trideseti", "triindvajseti",
     "triintrideseti", "trinajsti", "tristoti", "četrtima", "šestdesetima", "šestima", "šestnajstima",
     "štiridesetima",
     "štiriindvajsetima", "štirinajstima", "desetima", "devetdesetima", "devetima", "devetnajstima", "drugima",
     "dvaindevetdesetima", "dvajsetima", "dvanajstima", "dvestotima", "enaindvajsetima", "enajstima",
     "osemdesetima",
     "osemnajstima", "osmima", "petdesetima", "petima", "petinštiridesetima", "petindvajsetima",
     "petinosemdesetima",
     "petintridesetima", "petnajstima", "prvima", "sedemdesetima", "sedemindvajsetima", "sedemnajstima", "sedmima",
     "stotima", "tisočima", "tretjima", "tridesetima", "triindvajsetima", "triintridesetima", "trinajstima",
     "tristotima", "četrtih", "drugih", "dvaindevetdesetih", "prvih", "tretjih", "triintridesetih", "četrte",
     "šestdesete", "šeste", "šestnajste", "štiridesete", "štiriindvajsete", "štirinajste", "desete", "devetdesete",
     "devete", "devetnajste", "druge", "dvaindevetdesete", "dvajsete", "dvanajste", "dvestote", "enaindvajsete",
     "enajste", "osemdesete", "osemnajste", "osme", "petdesete", "pete", "petinštiridesete", "petindvajsete",
     "petinosemdesete", "petintridesete", "petnajste", "prve", "sedemdesete", "sedemindvajsete", "sedemnajste",
     "sedme",
     "stote", "tisoče", "tretje", "tridesete", "triindvajsete", "triintridesete", "trinajste", "tristote",
     "četrtim",
     "drugim", "dvaindevetdesetim", "prvim", "tretjim", "triintridesetim", "četrtimi", "drugimi",
     "dvaindevetdesetimi",
     "prvimi", "tretjimi", "triintridesetimi", "četrto", "šestdeseto", "šestnajsto", "šesto", "štirideseto",
     "štiriindvajseto", "štirinajsto", "deseto", "devetdeseto", "devetnajsto", "deveto", "drugo",
     "dvaindevetdeseto",
     "dvajseto", "dvanajsto", "dvestoto", "enaindvajseto", "enajsto", "osemdeseto", "osemnajsto", "osmo",
     "petdeseto",
     "petinštirideseto", "petindvajseto", "petinosemdeseto", "petintrideseto", "petnajsto", "peto", "prvo",
     "sedemdeseto", "sedemindvajseto", "sedemnajsto", "sedmo", "stoto", "tisočo", "tretjo", "trideseto",
     "triindvajseto", "triintrideseto", "trinajsto", "tristoto", "četrta", "šesta", "šestdeseta", "šestnajsta",
     "štirideseta", "štiriindvajseta", "štirinajsta", "deseta", "deveta", "devetdeseta", "devetnajsta", "druga",
     "dvaindevetdeseta", "dvajseta", "dvanajsta", "dvestota", "enaindvajseta", "enajsta", "osemdeseta",
     "osemnajsta",
     "osma", "peta", "petdeseta", "petinštirideseta", "petindvajseta", "petinosemdeseta", "petintrideseta",
     "petnajsta",
     "prva", "sedemdeseta", "sedemindvajseta", "sedemnajsta", "sedma", "stota", "tisoča", "tretja", "trideseta",
     "triindvajseta", "triintrideseta", "trinajsta", "tristota", "četrtega", "šestdesetega", "šestega",
     "šestnajstega",
     "štiridesetega", "štiriindvajsetega", "štirinajstega", "desetega", "devetdesetega", "devetega",
     "devetnajstega",
     "drugega", "dvaindevetdesetega", "dvajsetega", "dvanajstega", "dvestotega", "enaindvajsetega", "enajstega",
     "osemdesetega", "osemnajstega", "osmega", "petdesetega", "petega", "petinštiridesetega", "petindvajsetega",
     "petinosemdesetega", "petintridesetega", "petnajstega", "prvega", "sedemdesetega", "sedemindvajsetega",
     "sedemnajstega", "sedmega", "stotega", "tisočega", "tretjega", "tridesetega", "triindvajsetega",
     "triintridesetega", "trinajstega", "tristotega", "četrtemu", "šestdesetemu", "šestemu", "šestnajstemu",
     "štiridesetemu", "štiriindvajsetemu", "štirinajstemu", "desetemu", "devetdesetemu", "devetemu",
     "devetnajstemu",
     "drugemu", "dvaindevetdesetemu", "dvajsetemu", "dvanajstemu", "dvestotemu", "enaindvajsetemu", "enajstemu",
     "osemdesetemu", "osemnajstemu", "osmemu", "petdesetemu", "petemu", "petinštiridesetemu", "petindvajsetemu",
     "petinosemdesetemu", "petintridesetemu", "petnajstemu", "prvemu", "sedemdesetemu", "sedemindvajsetemu",
     "sedemnajstemu", "sedmemu", "stotemu", "tisočemu", "tretjemu", "tridesetemu", "triindvajsetemu",
     "triintridesetemu", "trinajstemu", "tristotemu", "četrtem", "šestdesetem", "šestem", "šestnajstem",
     "štiridesetem",
     "štiriindvajsetem", "štirinajstem", "desetem", "devetdesetem", "devetem", "devetnajstem", "drugem",
     "dvaindevetdesetem", "dvajsetem", "dvanajstem", "dvestotem", "enaindvajsetem", "enajstem", "osemdesetem",
     "osemnajstem", "osmem", "petdesetem", "petem", "petinštiridesetem", "petindvajsetem", "petinosemdesetem",
     "petintridesetem", "petnajstem", "prvem", "sedemdesetem", "sedemindvajsetem", "sedemnajstem", "sedmem",
     "stotem",
     "tisočem", "tretjem", "tridesetem", "triindvajsetem", "triintridesetem", "trinajstem", "tristotem", "deseteri",
     "dvakratni", "dvoji", "enkratni", "peteri", "stoteri", "tisočeri", "trikratni", "troji", "deseterima",
     "dvakratnima", "dvojima", "enkratnima", "peterima", "stoterima", "tisočerima", "trikratnima", "trojima",
     "deseterih", "dvakratnih", "dvojih", "enkratnih", "peterih", "stoterih", "tisočerih", "trikratnih", "trojih",
     "desetere", "dvakratne", "dvoje", "enkratne", "petere", "stotere", "tisočere", "trikratne", "troje",
     "deseterim",
     "dvakratnim", "dvojim", "enkratnim", "peterim", "stoterim", "tisočerim", "trikratnim", "trojim", "deseterimi",
     "dvakratnimi", "dvojimi", "enkratnimi", "peterimi", "stoterimi", "tisočerimi", "trikratnimi", "trojimi",
     "desetero", "dvakratno", "dvojo", "enkratno", "petero", "stotero", "tisočero", "trikratno", "trojo",
     "desetera",
     "dvakratna", "dvoja", "enkratna", "petera", "stotera", "tisočera", "trikratna", "troja", "deseterega",
     "dvakratnega", "dvojega", "enkratnega", "peterega", "stoterega", "tisočerega", "trikratnega", "trojega",
     "deseter",
     "dvakraten", "dvoj", "enkraten", "peter", "stoter", "tisočer", "trikraten", "troj", "deseteremu",
     "dvakratnemu",
     "dvojemu", "enkratnemu", "peteremu", "stoteremu", "tisočeremu", "trikratnemu", "trojemu", "deseterem",
     "dvakratnem", "dvojem", "enkratnem", "peterem", "stoterem", "tisočerem", "trikratnem", "trojem", "le-onega",
     "le-tega", "le-tistega", "le-toliko", "onega", "tega", "tistega", "toliko", "le-oni", "le-takšni", "le-taki",
     "le-te", "le-ti", "le-tisti", "oni", "takšni", "taki", "te", "ti", "tisti", "le-onima", "le-takšnima",
     "le-takima",
     "le-tema", "le-tistima", "onima", "takšnima", "takima", "tema", "tistima", "le-onih", "le-takšnih", "le-takih",
     "le-teh", "le-tistih", "onih", "takšnih", "takih", "teh", "tistih", "le-one", "le-takšne", "le-take",
     "le-tiste",
     "one", "takšne", "take", "tiste", "le-onim", "le-takšnim", "le-takim", "le-tem", "le-tistim", "onim",
     "takšnim",
     "takim", "tem", "tistim", "le-onimi", "le-takšnimi", "le-takimi", "le-temi", "le-tistimi", "onimi", "takšnimi",
     "takimi", "temi", "tistimi", "le-ono", "le-takšno", "le-tako", "le-tisto", "le-to", "ono", "takšno", "tako",
     "tisto", "to", "le-tej", "tej", "le-ona", "le-ta", "le-takšna", "le-taka", "le-tista", "ona", "ta", "takšna",
     "taka", "tista", "le-tak", "le-takšen", "tak", "takšen", "le-takšnega", "le-takega", "takšnega", "takega",
     "le-onemu", "le-takšnemu", "le-takemu", "le-temu", "le-tistemu", "onemu", "takšnemu", "takemu", "temu",
     "temuintemu", "tistemu", "le-onem", "le-takšnem", "le-takem", "le-tistem", "onem", "takšnem", "takem",
     "tistem",
     "vsakogar", "vsakomur", "vsakomer", "vsakdo", "obe", "vsaki", "vsakršni", "vsi", "obema", "vsakima",
     "vsakršnima",
     "vsema", "obeh", "vsakih", "vsakršnih", "vseh", "vsake", "vsakršne", "vse", "vsakim", "vsakršnim", "vsem",
     "vsakimi", "vsakršnimi", "vsemi", "vsako", "vsakršno", "vso", "vsej", "vsa", "vsaka", "vsakršna", "oba", "ves",
     "vsak", "vsakršen", "vsakega", "vsakršnega", "vsega", "vsakemu", "vsakršnemu", "vsemu", "vsakem", "vsakršnem",
     "enako", "istega", "koliko", "mnogo", "nekoga", "nekoliko", "precej", "kaj", "koga", "marsikaj", "marsikoga",
     "nekaj", "čemu", "komu", "marsičemu", "marsikomu", "nečemu", "nekomu", "česa", "marsičesa", "nečesa", "kom",
     "marsičim", "marsikom", "nečim", "nekom", "čem", "marsičem", "nečem", "kdo", "marsikdo", "nekdo", "čigavi",
     "drugačni", "enaki", "isti", "kakšni", "kaki", "kakršnikoli", "kateri", "katerikoli", "kolikšni", "koliki",
     "marsikateri", "nekakšni", "nekaki", "nekateri", "neki", "takile", "tele", "tile", "tolikšni", "toliki",
     "čigavima", "drugačnima", "enakima", "enima", "istima", "kakšnima", "kakima", "kakršnimakoli", "katerima",
     "katerimakoli", "kolikšnima", "kolikima", "marsikaterima", "nekakšnima", "nekakima", "nekaterima", "nekima",
     "takimale", "temale", "tolikšnima", "tolikima", "čigavih", "drugačnih", "enakih", "enih", "istih", "kakšnih",
     "kakih", "kakršnihkoli", "katerih", "katerihkoli", "kolikšnih", "kolikih", "marsikaterih", "nekakšnih",
     "nekakih",
     "nekaterih", "nekih", "takihle", "tehle", "tolikšnih", "tolikih", "čigave", "drugačne", "enake", "iste",
     "kakšne",
     "kake", "kakršnekoli", "katere", "katerekoli", "kolikšne", "kolike", "marsikatere", "nekakšne", "nekake",
     "nekatere", "neke", "takele", "tolikšne", "tolike", "čigavim", "drugačnim", "enakim", "istim", "kakšnim",
     "kakim",
     "kakršnimkoli", "katerim", "katerimkoli", "kolikšnim", "kolikim", "marsikaterim", "nekakšnim", "nekakim",
     "nekaterim", "nekim", "takimle", "temle", "tolikšnim", "tolikim", "čigavimi", "drugačnimi", "enakimi", "enimi",
     "istimi", "kakšnimi", "kakimi", "kakršnimikoli", "katerimi", "katerimikoli", "kolikšnimi", "kolikimi",
     "marsikaterimi", "nekakšnimi", "nekakimi", "nekaterimi", "nekimi", "takimile", "temile", "tolikšnimi",
     "tolikimi",
     "čigavo", "drugačno", "isto", "kakšno", "kako", "kakršnokoli", "katero", "katerokoli", "kolikšno",
     "marsikatero",
     "nekakšno", "nekako", "nekatero", "neko", "takole", "tole", "tolikšno", "tejle", "čigava", "drugačna", "enaka",
     "ista", "kakšna", "kaka", "kakršnakoli", "katera", "katerakoli", "kolikšna", "kolika", "marsikatera", "neka",
     "nekakšna", "nekaka", "nekatera", "takale", "tale", "tolikšna", "tolika", "čigav", "drug", "drugačen", "enak",
     "kak", "kakšen", "kakršenkoli", "kakršnegakoli", "kateregakoli", "kolik", "kolikšen", "nek", "nekak",
     "nekakšen",
     "takegale", "takle", "tegale", "tolik", "tolikšen", "čigavega", "drugačnega", "enakega", "kakšnega", "kakega",
     "katerega", "kolikšnega", "kolikega", "marsikaterega", "nekakšnega", "nekakega", "nekaterega", "nekega",
     "tolikšnega", "tolikega", "čigavemu", "drugačnemu", "enakemu", "istemu", "kakšnemu", "kakemu", "kakršnemukoli",
     "kateremu", "kateremukoli", "kolikšnemu", "kolikemu", "marsikateremu", "nekakšnemu", "nekakemu", "nekateremu",
     "nekemu", "takemule", "temule", "tolikšnemu", "tolikemu", "čigavem", "drugačnem", "enakem", "istem", "kakšnem",
     "kakem", "kakršnemkoli", "katerem", "kateremkoli", "kolikšnem", "kolikem", "marsikaterem", "nekakšnem",
     "nekakem",
     "nekaterem", "nekem", "takemle", "tolikšnem", "tolikem", "naju", "nama", "midva", "nas", "nam", "nami", "mi",
     "mene", "me", "meni", "mano", "menoj", "jaz", "vaju", "vama", "vidva", "vas", "vam", "vami", "vi", "tebe",
     "tebi",
     "tabo", "teboj", "njiju", "jih", "ju", "njima", "jima", "onedve", "onidve", "nje", "njih", "njim", "jim",
     "njimi",
     "njo", "jo", "njej", "nji", "ji", "je", "onadva", "njega", "ga", "njemu", "mu", "njem", "on", "čigar",
     "kolikor",
     "kar", "karkoli", "kogar", "kogarkoli", "čemur", "čemurkoli", "komur", "komurkoli", "česar", "česarkoli",
     "čimer",
     "čimerkoli", "komer", "komerkoli", "čemer", "čemerkoli", "kdor", "kdorkoli", "kakršni", "kakršnima",
     "kakršnih",
     "kakršne", "kakršnim", "kakršnimi", "kakršno", "kakršna", "kakršen", "kakršnega", "kakršnemu", "kakršnem",
     "najini", "naši", "moji", "najinima", "našima", "mojima", "najinih", "naših", "mojih", "najine", "naše",
     "moje",
     "najinim", "našim", "mojim", "najinimi", "našimi", "mojimi", "najino", "našo", "mojo", "najina", "naša",
     "moja",
     "najin", "najinega", "naš", "našega", "moj", "mojega", "najinemu", "našemu", "mojemu", "najinem", "našem",
     "mojem",
     "vajini", "vaši", "tvoji", "vajinima", "vašima", "tvojima", "vajinih", "vaših", "tvojih", "vajine", "vaše",
     "tvoje", "vajinim", "vašim", "tvojim", "vajinimi", "vašimi", "tvojimi", "vajino", "vašo", "tvojo", "vajina",
     "vaša", "tvoja", "vajin", "vajinega", "vaš", "vašega", "tvoj", "tvojega", "vajinemu", "vašemu", "tvojemu",
     "vajinem", "vašem", "tvojem", "njuni", "njihovi", "njeni", "njegovi", "njunima", "njihovima", "njenima",
     "njegovima", "njunih", "njihovih", "njenih", "njegovih", "njune", "njihove", "njene", "njegove", "njunim",
     "njihovim", "njenim", "njegovim", "njunimi", "njihovimi", "njenimi", "njegovimi", "njuno", "njihovo", "njeno",
     "njegovo", "njuna", "njihova", "njena", "njegova", "njun", "njunega", "njihov", "njihovega", "njen", "njenega",
     "njegov", "njegovega", "njunemu", "njihovemu", "njenemu", "njegovemu", "njunem", "njihovem", "njenem",
     "njegovem",
     "se", "si", "sebe", "sebi", "sabo", "seboj", "svoji", "svojima", "svojih", "svoje", "svojim", "svojimi",
     "svojo",
     "svoja", "svoj", "svojega", "svojemu", "svojem", "nikogar", "noben", "ničemur", "nikomur", "ničesar",
     "ničimer",
     "nikomer", "ničemer", "nihče", "nikakršni", "nobeni", "nikakršnima", "nobenima", "nikakršnih", "nobenih",
     "nikakršne", "nobene", "nikakršnim", "nobenim", "nikakršnimi", "nobenimi", "nikakršno", "nobeno", "nikakršna",
     "nobena", "nikakršen", "nikakršnega", "nobenega", "nikakršnemu", "nobenemu", "nikakršnem", "nobenem", "še",
     "šele",
     "žal", "že", "baje", "bojda", "bržčas", "bržkone", "celo", "dobesedno", "domala", "edinole", "gotovo", "itak",
     "ja", "kajne", "kajpada", "kajpak", "koli", "komaj", "le", "malone", "mar", "menda", "morda", "morebiti",
     "nadvse",
     "najbrž", "nemara", "nerad", "neradi", "nikar", "pač", "pogodu", "prav", "pravzaprav", "predvsem", "preprosto",
     "rad", "rada", "rade", "radi", "ravno", "res", "resda", "samo", "seveda", "skoraj", "skorajda", "spet",
     "sploh",
     "tudi", "všeč", "verjetno", "vnovič", "vred", "vsaj", "zadosti", "zapored", "zares", "zgolj", "zlasti",
     "zopet",
     "čezenj", "čeznje", "mednje", "mednju", "medse", "nadenj", "nadme", "nadnje", "name", "nanj", "nanje", "nanjo",
     "nanju", "nase", "nate", "obenj", "podnjo", "pome", "ponj", "ponje", "ponjo", "pote", "predenj", "predme",
     "prednje", "predse", "skozenj", "skoznje", "skoznjo", "skozte", "vame", "vanj", "vanje", "vanjo", "vanju",
     "vase",
     "vate", "zame", "zanj", "zanje", "zanjo", "zanju", "zase", "zate", "čez", "med", "na", "nad", "ob", "po",
     "pod",
     "pred", "raz", "skoz", "skozi", "v", "za", "zoper", "h", "k", "kljub", "nasproti", "navkljub", "navzlic",
     "proti",
     "ž", "blizu", "brez", "dno", "do", "iz", "izmed", "iznad", "izpod", "izpred", "izven", "izza", "krog", "mimo",
     "namesto", "naokoli", "naproti", "od", "okoli", "okrog", "onkraj", "onstran", "poleg", "povrh", "povrhu",
     "prek",
     "preko", "razen", "s", "spod", "spričo", "sredi", "vštric", "vpričo", "vrh", "vrhu", "vzdolž", "z", "zaradi",
     "zavoljo", "zraven", "zunaj", "o", "pri", "bi", "bova", "bomo", "bom", "bosta", "boste", "boš", "bodo", "bojo",
     "bo", "sva", "nisva", "smo", "nismo", "sem", "nisem", "sta", "nista", "ste", "niste", "nisi", "so", "niso",
     "ni",
     "bodiva", "bodimo", "bodita", "bodite", "bodi", "biti", "bili", "bila", "bile", "bil", "bilo", "želiva",
     "dovoliva", "hočeva", "marava", "morava", "moreva", "smeva", "zmoreva", "nočeva", "želimo", "dovolimo",
     "hočemo",
     "maramo", "moramo", "moremo", "smemo", "zmoremo", "nočemo", "želim", "dovolim", "hočem", "maram", "moram",
     "morem",
     "smem", "zmorem", "nočem", "želita", "dovolita", "hočeta", "marata", "morata", "moreta", "smeta", "zmoreta",
     "nočeta", "želite", "dovolite", "hočete", "marate", "morate", "morete", "smete", "zmorete", "nočete", "želiš",
     "dovoliš", "hočeš", "maraš", "moraš", "moreš", "smeš", "zmoreš", "nočeš", "želijo", "dovolijo", "hočejo",
     "marajo",
     "morajo", "morejo", "smejo", "zmorejo", "nočejo", "želi", "dovoli", "hoče", "mara", "mora", "more", "sme",
     "zmore",
     "noče", "hotiva", "marajva", "hotimo", "marajmo", "hotita", "marajta", "hotite", "marajte", "hoti", "maraj",
     "želeti", "dovoliti", "hoteti", "marati", "moči", "morati", "smeti", "zmoči", "želeni", "dovoljeni", "želena",
     "dovoljena", "želene", "dovoljene", "želen", "dovoljen", "želeno", "dovoljeno", "želeli", "dovolili", "hoteli",
     "marali", "mogli", "morali", "smeli", "zmogli", "želela", "dovolila", "hotela", "marala", "mogla", "morala",
     "smela", "zmogla", "želele", "dovolile", "hotele", "marale", "mogle", "morale", "smele", "zmogle", "želel",
     "dovolil", "hotel", "maral", "mogel", "moral", "smel", "zmogel", "želelo", "dovolilo", "hotelo", "maralo",
     "moglo",
     "moralo", "smelo", "zmogl"])


class TextUtils(object):

    def __init__(self, stop_words):
        self.stop_words = SLOVENE_STOP_WORDS

    pattern = r"(?u)\b\w\w+\b"

    lemmatizer = WordNetLemmatizer()

    punc = list(set(string.punctuation))

    def casual_tokenizer(self, text):
        tokenizer = TweetTokenizer()
        tokens = tokenizer.tokenize(text)
        return tokens

    def slovene_lematize(self, tagger):
        lemmatizer = Lemmatizer(dictionary=lemmagen.DICTIONARY_SLOVENE)
        lematized = []
        for tag in tagger:
            word = tag[0]
            lemma = lemmatizer.lemmatize(word)
            lematized.append(lemma)
        return lematized

    def process_text(self, text):
        punc = list(set(string.punctuation))
        soup = BeautifulSoup(text, "lxml")
        tags_del = soup.get_text()
        no_html = re.sub('<[^>]*>', '', tags_del)
        tokenized = self.casual_tokenizer(no_html)
        lower = [item.lower() for item in tokenized]
        tagged = nltk.pos_tag(lower)
        lemma = self.slovene_lematize(tagged)
        no_num = [re.sub('[0-9]+', '', each) for each in lemma]
        no_punc = [w for w in no_num if w not in punc]
        no_stop = [w for w in no_punc if w not in self.stop_words]
        return no_stop


class Plotting(object):
    def __init__(self):
        pass

    def get_data_non_binarized(self, X, y):
        # Train test split with stratified sampling. Using non-binarized labels
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=.3,
                                                            shuffle=True,
                                                            stratify=y,
                                                            random_state=3)
        return X_train, X_test, y_train, y_test

    def get_data_binarized(self, X, y, class_names):
        # Binarize the labels
        y_b = label_binarize(y, classes=[i for i in range(len(class_names))])

        # Shuffle and split training and test sets with stratified sampling and binarized labels
        X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X,
                                                                    y_b,
                                                                    test_size=.3,
                                                                    shuffle=True,
                                                                    stratify=y,
                                                                    random_state=3)
        return X_train_b, X_test_b, y_train_b, y_test_b

    def plot_ROC(self, X, y, class_names, best_model):

        n_classes = len(class_names)
        X_train_b, X_test_b, y_train_b, y_test_b = self.get_data_binarized(X, y, class_names)

        # Learn to predict each class against the other
        classifier = OneVsRestClassifier(best_model)

        y_score = classifier.fit(X_train_b, y_train_b).predict_proba(X_test_b)

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_b[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr['micro'], tpr['micro'], _ = roc_curve(y_test_b.ravel(), y_score.ravel())
        roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr['macro'] = all_fpr
        tpr['macro'] = mean_tpr
        roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])

        # Plot all ROC curves
        plt.figure(figsize=(13, 10))
        sns.set_style('darkgrid')
        lw = 2

        plt.plot(fpr['micro'],
                 tpr['micro'],
                 label='micro-average ROC curve (area = {0:0.3f})'''.format(roc_auc['micro']),
                 color='deeppink',
                 linestyle=':',
                 linewidth=4)

        plt.plot(fpr['macro'],
                 tpr['macro'],
                 label='macro-average ROC curve (area = {0:0.3f})'''.format(roc_auc['macro']),
                 color='navy',
                 linestyle=':',
                 linewidth=4)

        colors = cycle(['#41924F', '#FFC300', '#a98ff3', '#59C7EA', '#E9474A'])

        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i],
                     color=color,
                     lw=lw,
                     label='ROC curve of class {0} (area = {1:0.3f})'''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('1-Specificity (False Positive Rate)', fontsize=14)
        plt.ylabel('Sensitivity (True Positive Rate)', fontsize=14)
        plt.title('Receiver Operating Characteristic', fontsize=16)
        plt.legend(loc="lower right", fontsize=13)
        plt.savefig(model_path + 'ROC_CURVE.png')
        plt.show()

    def create_confusion_matrix(self, y_test, y_pred):
        # Get the confusion matrix and put it into a df
        cm = confusion_matrix(y_test, y_pred)

        cm_df = pd.DataFrame(cm,
                             index=['avto', 'sport', 'svet', 'slovenija', 'magazin'],
                             columns=['avto', 'sport', 'svet', 'slovenija', 'magazin'])

        # Plot the heatmap
        plt.figure(figsize=(12, 8))

        sns.heatmap(cm_df,
                    center=0,
                    cmap=sns.diverging_palette(220, 15, as_cmap=True),
                    annot=True,
                    fmt='g')

        plt.title(
            'F1 Score (avg = macro) : {0:.2f}'.format(f1_score(y_test, y_pred, average='macro')),
            fontsize=13)
        plt.ylabel('True label', fontsize=13)
        plt.xlabel('Predicted label', fontsize=13)
        plt.savefig(model_path + 'Conf_matrix.png')
        plt.show()


class DeepModels(object):

    def __init__(self, data):
        self.data = data  # data object

    def ensure_dir(self, file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    def create_embedding_models(self, vocab_size, max_len, output_sizes, n_denses):
        models_ = []
        for n_dense in n_denses:
            for output_size in output_sizes:
                model_, description = self.create_embeding_model(max_len, vocab_size, output_size, n_dense)
                models_.append((model_, description))
                return models_

    def create_sequential_models(self, max_len, output_shapes):
        models_ = []
        for output_shape in output_shapes:
            model_, description = self.create_sequential_model(max_len, output_shape)
            models_.append((model_, description))
        return models_

    def create_cnn_models(self, MAX_LENGTH, vocab_size, embedding_dims, conv_sizes, hiddens):
        models_ = []
        for embedding_dim in embedding_dims:
            for conv_size in conv_sizes:
                for hidden in hiddens:
                    model_, description = self.create_cnn(MAX_LENGTH, vocab_size, embedding_dim, conv_size, hidden)
                    models_.append((model_, description))
        return models_

    def create_lstm_models(self, MAX_LENGTH, vocab_size, embedding_dims, lstm_layers, hiddens):
        models_ = []
        for embedding_dim in embedding_dims:
            for lstm_layer in lstm_layers:
                for hidden_layer in hiddens:
                    model_, description = self.create_LSTM(MAX_LENGTH, vocab_size, embedding_dim, lstm_layer,
                                                           hidden_layer)
                    models_.append((model_, description))
        return models_

    def create_embeding_model(self, MAX_LENGTH, vocab_size, outputsize, n_dense):
        inputs = layers.Input(shape=(MAX_LENGTH,))
        embedding_layer = layers.Embedding(vocab_size, outputsize, input_length=MAX_LENGTH)(inputs)
        x = layers.Flatten()(embedding_layer)
        x = layers.Dense(n_dense, activation='relu')(x)
        predictions = layers.Dense(5, activation='softmax')(x)
        model = models.Model(inputs=[inputs], outputs=predictions)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        model_params = "Embedding model: layer input:{} vocab_size {} output: {} dense: {} ".format(MAX_LENGTH,
                                                                                                    vocab_size,
                                                                                                    outputsize, n_dense)
        return model, model_params

    def create_sequential_model(self, MAX_LENGTH, output_shape):
        max_words = MAX_LENGTH
        num_classes = 5
        # Build the model
        model = models.Sequential()
        model.add(layers.Dense(output_shape, input_shape=(max_words,)))
        model.add(layers.Activation('relu'))
        # model.add(layers.Dropout(drop_ratio))
        model.add(layers.Dense(num_classes))
        model.add(layers.Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['acc'])
        model_params = "Seq model: layer input:{} output: {}".format(MAX_LENGTH, output_shape)
        return model, model_params

    def create_cnn(self, MAX_LENGTH, vocab_size, embedding_dim, conv_size, hidden):  # CNN
        model = models.Sequential()
        model.add(layers.Embedding(vocab_size, embedding_dim, input_length=MAX_LENGTH))
        model.add(layers.Conv1D(conv_size, 5, activation='relu'))
        model.add(layers.GlobalMaxPooling1D())
        model.add(layers.Dense(hidden, activation='relu'))
        model.add(layers.Dense(5, activation='softmax'))
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model_params = "Cnn: embeding: {} layer input_length:{} vocab_size {} conv_size: {} hidden: {} " \
            .format(embedding_dim, MAX_LENGTH, vocab_size, conv_size, hidden)
        return model, model_params

    def create_LSTM(self, MAX_LENGTH, vocab_size, embedding_dim, lstm_layer, hidden_layer):
        inputs = layers.Input(shape=(MAX_LENGTH,))
        embedding_layer = layers.Embedding(vocab_size, embedding_dim, input_length=MAX_LENGTH)(inputs)
        x = layers.LSTM(lstm_layer)(embedding_layer)
        x = layers.Dense(hidden_layer, activation='relu')(x)
        predictions = layers.Dense(5, activation='softmax')(x)
        model = models.Model(inputs=[inputs], outputs=predictions)
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model_params = "lstm : embeding: {} layer input_length:{} vocab_size {} lstm_layer: {} hidden: {} " \
            .format(embedding_dim, MAX_LENGTH, vocab_size, lstm_layer, hidden_layer)
        return model, model_params

    def create_experiment(self):
        data = self.data
        print("==== Step 1: Generating dataset...")
        logging.info("start")
        data.generate_dataset_from_file(global_filename)
        print("==== Step 2: Loading dataset...")
        df = data.load_dataset(dataset_path)
        class_names = df['label'].unique()
        init_csv = model_path + 'init.csv'
        df.to_csv(init_csv)
        print("class names:", class_names)
        # print ("df head:", df.head(5))

        print("==== Step 3: Preporcesing texts...")
        # Apply the function to preprocess the text. Tokenize, lower, expand contactions, lemmatize, remove punctuation, numbers and stop words
        df['clean_text'] = df['content'].apply(utils.process_text)
        # pipes_settings = get_all(data, df)
        print("==== Step 5: Splitting data...")
        LE = LabelEncoder()
        df['label_num'] = LE.fit_transform(df['label'])
        X = df['clean_text'].astype('str')
        y = df['label_num'].values

        tokenizer = keras.preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(X.values)
        post_seq = tokenizer.texts_to_sequences(X.values)
        vocab_size = len(tokenizer.word_index) + 1

        # TODO: GridSearch does not work with Keras !!!!
        # Same problem with saving models
        # https://github.com/tensorflow/tensorflow/issues/33204

        best_acc, curr_best_model = 0.0, None
        for max_len in [500, 1000, 10000]:
            post_seq_padded = keras.preprocessing.sequence.pad_sequences(post_seq, maxlen=max_len)
            embeding_models = self.create_embedding_models(vocab_size, max_len, [128, 248], [32, 64])
            # seq_models = self.create_sequential_models(max_len, [512, 1024])
            cnn_models = self.create_cnn_models(max_len, vocab_size, [100, 1000], [128, 256], [10, 100, 500])
            lstm_models = self.create_lstm_models(max_len, vocab_size, [100, 1000], [64, 128], [10, 100])

            check_models = embeding_models + cnn_models + lstm_models
            X_train, X_test, y_train, y_test = train_test_split(post_seq_padded, y, test_size=0.3, random_state=1000)

            for i, model_c in enumerate(check_models):
                print("model:", len(model_c))
                model = model_c[0]
                print("\n-----------------------")
                print("{}/{}".format(i + 1, len(check_models)))
                print(model_c[1])  # description
                batch_sizes = [32, 64, 128]
                for batch_size in batch_sizes:
                    print("max len:", max_len, "batch_size:", batch_size)
                    filepath = "dnn_" + str(i) + "_" + str(max_len) + "_" + str(batch_size) + ".mod"
                    checkpointer = keras.callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=0,
                                                                   save_best_only=True,
                                                                   mode='min')
                    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0,
                                                                   verbose=0,
                                                                   mode='auto', baseline=None,
                                                                   restore_best_weights=True)

                    history = model.fit([X_train], batch_size=batch_size, y=keras.utils.to_categorical(y_train),
                                        verbose=1,
                                        validation_split=0.25,
                                        shuffle=True, epochs=30, callbacks=[checkpointer, early_stopping])

                    predicted = model.predict(X_test)
                    predicted = np.argmax(predicted, axis=1)
                    score = accuracy_score(y_test, predicted)
                    if score > best_acc:
                        best_acc = score
                        curr_best_model = model
                        # self.save_model(model_path, "curr_best_deep_model.mod", curr_best_model)
                    print(score)

                    print("******************************")

        print("final score {} best of the best:", best_acc, curr_best_model)
        # print ("params:", curr_best_model.get_params())
        curr_best_model.fit(X, y)
        score = curr_best_model.score(X_test, y_test)

        print("score on test set:", score)
        # self.save_model(model_path, "the_best_deep.mod", curr_best_model)

    def predict_url(self, url):
        data = Data()
        tu = TextUtils(SLOVENE_STOP_WORDS)
        content = data.get_content(url)
        df = pd.DataFrame(columns=['content'], data=[content])
        df['clean_text'] = df['content'].apply(tu.process_text)
        unseen = df['clean_text'].astype('str')
        model = data.load_model(model_path, 'the_best3.mod')
        pred = model.predict(unseen)[0]
        pred_proba = model.predict_proba(unseen)[0][pred]
        label_pred = classnames_dict[pred]
        print("pred:", label_pred, "probability:", pred_proba)


class SciKitModels(object):

    def __init__(self, data):
        self.data = data  # data object

    def get_model_params(self, dict_model):
        random_state = [3]
        clf_name = dict_model['name']
        if clf_name is 'SGDClassifier':
            clf = SGDClassifier()
            params = dict(clf__loss=dict_model['params']['loss'],
                          clf__penalty=dict_model['params']['penalty'],
                          clf__alpha=dict_model['params']['alpha'],
                          clf__max_iter=dict_model['params']['max_iter'],
                          clf__tol=dict_model['params']['tol'],
                          clf__random_state=random_state,
                          clf__early_stopping=[True])
        elif clf_name is 'RandomForestClassifier':
            clf = RandomForestClassifier()
            params = dict(clf__bootstrap=dict_model['params']['bootstrap'],
                          clf__max_depth=dict_model['params']['max_depth'],
                          clf__max_features=dict_model['params']['max_features'],
                          clf__min_samples_leaf=dict_model['params']['min_samples_leaf'],
                          clf__n_estimators=dict_model['params']['n_estimators'],
                          clf__random_state=random_state)
        elif clf_name is 'LogisticRegression':
            clf = LogisticRegression()
            params = dict(clf__C=dict_model['params']['C'],
                          clf__penalty=dict_model['params']['penalty'],
                          clf__max_iter=dict_model['params']['max_iter'],
                          clf__tol=dict_model['params']['tol'],
                          clf__solver=dict_model['params']['solver'],
                          clf__multi_class=dict_model['params']['multi_class'],
                          clf__random_state=random_state)
        return clf, params

    def get_features(self, df, feature_type):
        feature_path = model_path + "features.csv"
        df.to_csv(feature_path)
        feature_create, feature_name = feature_type['feature'], feature_type['name']
        if feature_name == 'TfidfVectorizer':  # n-gram level features
            if feature_type['level'] == 'n-gram':
                print("tdif n-gram")
                # vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=.95, max_features=1000)
                vectorizer = TfidfVectorizer()
                parameters = {'feature__max_df': [1.0],
                              'feature__min_df': [2, 3],
                              'feature__ngram_range': [(1, 2)],
                              }
                """
                parameters = {'feature__max_df': [0.25, 0.5, 0.75, 1.0],
                              'feature__min_df':[2,3],
                              'feature__ngram_range':[(1, 2), (2, 3)],
                              'feature__smooth_idf': [True, False],
                              'feature__norm': ['l1', 'l2', None],
                              'feature__max_features': [1000, 5000, 10000],
                              }
                """
                # self.save_model(model_path, 'n-gram.vec', vectorizer)
            elif feature_type['level'] == 'word_char':  # word level td-idf
                print("tdif word")
                vectorizer = TfidfVectorizer(token_pattern=r'\w{1,}')
                parameters = {'feature__analyzer': ['word'],
                              'feature__max_df': [1.0],
                              'feature__min_df': [2],
                              'feature__ngram_range': [(1, 2)],
                              'feature__norm': ['l1'],
                              'feature__max_features': [1000],
                              }
                """
                parameters = {'feature__analyzer': ['word', 'char'],
                              'feature__max_df': [0.25, 0.5, 0.75, 1.0],
                             'feature__min_df': [2, 3],
                             'feature__ngram_range': [(1, 2), (2, 3)],
                             'feature__smooth_idf': [True, False],
                             'feature__norm': ['l1', 'l2', None],
                             'feature__max_features': [1000, 5000, 10000],
                             }
                """
        elif feature_name == 'CountVectorizer':
            print("count word")
            vectorizer = CountVectorizer(token_pattern=r'\w{1,}')
            parameters = {'feature__analyzer': ['char'],
                          'feature__max_df': [1.0],
                          'feature__min_df': [2],
                          'feature__ngram_range': [(1, 2)],
                          'feature__max_features': [1000],
                          }
            """
            parameters = {'feature__analyzer': ['char', 'word'],
                          'feature__max_df': [0.25, 0.5, 0.75, 1.0],
                          'feature__min_df': [2, 3],
                          'feature__ngram_range': [(1, 2), (2, 3)],
                          'feature__max_features': [1000, 5000, 10000],
                          }
            """

        return vectorizer, parameters

    def get_step1_models(self, data_object, df):
        pipes = []
        for fmodel in final_models:
            for feature_ in fetures_dict:
                # First add without sampling
                clf, param_dict = self.get_model_params(fmodel)
                vectorizer, parameters = self.get_features(df, feature_)
                param_dict = {**param_dict, **parameters}
                pipe = Pipeline(steps=[('feature', vectorizer), ('clf', clf)])  # tale dela
                insert_tuple = [pipe, param_dict]
                pipes.append(insert_tuple)
        return pipes

    def get_step2_models(self, best_1_pipeline):
        pipes = []
        for sampler_item in sampling_dict:
            sampler = sampler_item['method']
            pipe = imbPipeline(steps=[('feature', best_1_pipeline['feature']), ('sampling', sampler),
                                      ('clf', best_1_pipeline['clf'])])  # tale dela
            insert_tuple = [pipe, {}]
            pipes.append(insert_tuple)
        return pipes

    def get_step3_models(self, best_2_pipeline, sampling_better=False):
        pipes = []
        for dim_red_item in reduction_dict:
            dim_red = dim_red_item['method']
            if sampling_better is True:
                pipe = imbPipeline(steps=[('feature', best_2_pipeline['feature']),
                                          ('sampling', best_2_pipeline['sampling']),
                                          ('dim_red', dim_red),
                                          ('clf', best_2_pipeline['clf'])])  # tale dela
            else:  # Maybe sampling produces worst result
                pipe = Pipeline(steps=[('feature', best_2_pipeline['feature']),
                                       ('dim_red', dim_red),
                                       ('clf', best_2_pipeline['clf'])])  # tale dela

            insert_tuple = [pipe, {}]
            pipes.append(insert_tuple)
        return pipes

    def get_best_model(self, pipes_settings, data):
        X_train, X_test, y_train, y_test = data[0], data[1], data[2], data[3]
        models_params, model_scores = [], []
        print("pipe lens:", len(pipes_settings))
        for i, pipe_tuple in enumerate(pipes_settings):
            print("{}/{}".format(str(i + 1), len(pipes_settings)))
            pipe = pipe_tuple[0]
            # print("pipe: {}".format(pipe))
            logging.info("pipe: {}".format(str(pipe)))
            # print ("pipe:", pipe)
            param_dict = pipe_tuple[1]
            print("param_dict: {}", format(param_dict))
            print("==== Step 6: Grid search...")
            estimator = GridSearchCV(pipe, param_dict, cv=5, verbose=1, scoring='accuracy', n_jobs=-1)
            estimator.fit(X_train, y_train)
            best_estimator = estimator.best_estimator_
            score = estimator.score(X_test, y_test)
            print("test score: {} ".format(score))
            logging.info("test score: {} ".format(score))
            model_scores.append(score)
            models_params.append(best_estimator)

        # Save the best model
        best_idx = np.argmax(model_scores)
        print("best idx:", best_idx)
        best_pipeline = models_params[best_idx]
        print("best params:", best_pipeline)
        score = best_pipeline.score(X_test, y_test)
        print("score on test set:", score)
        return best_pipeline, score

    def create_experiment(self, utils_object):
        data = Data()
        print("==== Step 1: Generating dataset...")
        logging.info("start")
        data.generate_dataset_from_file(global_filename)
        print("==== Step 2: Loading dataset...")
        df = data.load_dataset(dataset_path)
        class_names = df['label'].unique()
        init_csv = model_path + 'init.csv'
        df.to_csv(init_csv)
        print("class names:", class_names)
        # print ("df head:", df.head(5))
        print("==== Step 3: Preporcesing texts...")
        # Apply the function to preprocess the text. Tokenize, lower, expand contactions, lemmatize, remove punctuation, numbers and stop words
        df['clean_text'] = df['content'].apply(utils_object.process_text)
        # pipes_settings = get_all(data, df)

        pipes_settings = self.get_step1_models(data, df)
        print("==== Step 5: Splitting data...")
        X = df['clean_text'].astype('str')
        # print ("df['label']:", df['label'])
        y = df['label'].values.astype('int')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, shuffle=True, stratify=y,
                                                            random_state=3)

        print("Xtrain:", X_train.shape)
        print("X_test:", X_test.shape)
        print("y_train:", y_train.shape)
        print("y_test:", y_test.shape)
        data_input = [X_train, X_test, y_train, y_test]

        best_model_pipeline, score = self.get_best_model(pipes_settings, data_input)
        best = score
        print("best sample:", best_model_pipeline)
        data.save_model(model_path, "the_best1.mod", best_model_pipeline)
        print("step 1 finished")

        best_step2_model, best_step3_model = None, None
        pipes_settings = self.get_step2_models(best_model_pipeline)
        best_sampling_pipeline, score = self.get_best_model(pipes_settings, data_input)
        sampling_better = False
        if score > best:
            best = score
            best_step2_model = best_sampling_pipeline
            sampling_better = True
            print("best sample:", best_sampling_pipeline)
            logging.info("best new score from step2:".format(score))
        else:
            best_step2_model = best_model_pipeline
        data.save_model(model_path, "the_best2.mod", best_step2_model)
        print("step 2 finished")

        pipes_settings = self.get_step3_models(best_step2_model, sampling_better)
        best_dim_red_pipeline, score = self.get_best_model(pipes_settings, data_input)
        if score > best:
            best = score
            best_step3_model = best_dim_red_pipeline
            logging.info("best new score from step3:".format(score))
            print("best sample:", best_dim_red_pipeline)
            print("step 3 finished")
        else:
            best_step3_model = best_step2_model

        print("best of the best:", best_step3_model)
        best_step3_model.fit(X, y)
        score = best_step3_model.score(X_test, y_test)
        print("score on test set:", score)
        data.save_model(model_path, "the_best3.mod", best_step3_model)

        # Predict the testing data
        y_pred = best_step3_model.predict(X_test)
        plotting = Plotting()
        plotting.plot_ROC(X, y, class_names, best_step3_model)
        plotting.create_confusion_matrix(y_test, y_pred)

    def predict_url(self, url):
        data = Data()
        tu = TextUtils(SLOVENE_STOP_WORDS)
        content = data.get_content(url)
        df = pd.DataFrame(columns=['content'], data=[content])
        df['clean_text'] = df['content'].apply(tu.process_text)
        unseen = df['clean_text'].astype('str')
        model = data.load_model(model_path, 'the_best3.mod')
        pred = model.predict(unseen)[0]
        pred_proba = model.predict_proba(unseen)[0][pred]
        label_pred = classnames_dict[pred]
        print("pred:", label_pred, "probability:", pred_proba)


if __name__ == '__main__':

    print(len(sys.argv))
    if len(sys.argv) == 1:
        data = Data()
        tu = TextUtils(SLOVENE_STOP_WORDS)
        # sci-kit models
        sci_kit = SciKitModels(data)
        sci_kit.create_experiment(tu)

        # tensor flow models
        # dnn = DeepModels(data)
        # dnn.create_experiment()
    elif len(sys.argv) == 2:
        url = str(sys.argv[1])
        if 'http' in url:
            data = Data()
            sci_kit = SciKitModels(data)
            sci_kit.predict_url(url)
            # dnn = DeepModels(data)
            # dnn.predict_url(url)
        else:
            raise Exception("url {} is not in http format".format(url))

    """
    if sys.argv[0]:
        filepath = sys.argv[1]
    else:
        filepath = global_filename
    """
