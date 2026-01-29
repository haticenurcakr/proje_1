"""
# PROJE: RNN ile duygu analizi (Sentiment Analysis)
# PROBLEM: Bir ürün yorumunun olumlu (1) ya da olumsuz (0) olup olmadığını tahmin etmek
# PROBLEM TÜRÜ: Metin tabanlı ikili sınıflandırma (Binary Text Classification)

# VERİ SETİ: Amazon ürün yorumları
# ETİKETLER: 0 = Negatif, 1 = Pozitif

# MODEL: RNN (Recurrent Neural Network)
# AÇIKLAMA: RNN'ler metin gibi sıralı verilerde önceki kelimelerden öğrenilen bağlamı
# sonraki kelimelerin yorumlanmasında kullanır

# AMAÇ: Kullanıcı yorumlarının genel duygu durumunu doğru şekilde tahmin etmek

# EK ANALİZ: Yorumlarda geçen ana konuların (kamera, batarya, fiyat vb.)
# belirlenmesi ve e-ticaret filtreleme senaryosuna uygun veri üretimi


Plan/program: 


Gerekli kurulumlar:


import lib:

"""
# import lab 

# -------------------- KÜTÜPHANELER --------------------

import numpy as np                      # Sayısal işlemler için
import pandas as pd                     # CSV veri setini okumak için
import nltk                             # Metin işleme araçları
from nltk.corpus import stopwords       # Gereksiz kelimeleri (the, is, vs.) temizlemek için
import matplotlib.pyplot as plt          # Grafik ve görselleştirme için

# -------------------- TENSORFLOW / KERAS --------------------

from tensorflow.keras.models import Sequential   # Sıralı (Sequential) model yapısı
from tensorflow.keras.layers import (
    Embedding,     # Kelimeleri sayısal vektörlere dönüştürür
    SimpleRNN,     # RNN katmanı (metin gibi sıralı veriler için)
    Dense          # Çıkış katmanı (olumlu / olumsuz)
)

from tensorflow.keras.preprocessing.text import Tokenizer # Metni (yorumları) sayısal dizilere çevirmek için
from tensorflow.keras.preprocessing.sequence import pad_sequences # Yorumların uzunluklarını eşitlemek için (RNN sabit uzunluk ister)


# stop word listesi

# load data set

#örnek veri incelemesi

#ön işleme veri temizleme

#dataa prepossing 

#RNN modeli oluşturulması

#traninig

#evolation

