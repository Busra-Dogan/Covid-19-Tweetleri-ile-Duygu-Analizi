# Sentiment Analysis with Covid19 Tweets
Bitirme Tasarımı çalışması olarak sunulmuştur ve kabul edilmiştir. 
Covid-19 etiketiyle atılan tweetlerin pozitif ve negatif olarak duygu analizi gerçekleştirilmiştir. Çalışma Python dili ve Anaconda Spyder üzerinde gerçekleştirilmiştir.

Veri Seti Oluşturma:
Öncelikle tweet içeriğinde gerekli alanlar belirlenmiştir. Ardından tweet veri kaynakları araştırılmıştır. Hazır veri seti sunan bir web sitesinden twweet verileri elde edilmiştir. csv dosyası incelenip 730 tane tweet manuel olarak pozitif(1) ve negatif(0) etiketleriyle etiketlenmiştir.

Veri Ön İşleme:
Elde edilen verilerde eksik alanlar, tekrarlanan veriler gibi istenmeyen durumlar söz konusu olabilmektedir. Bu çalışmada veri seti üzerinde istenmeyen karakterlerin ve kullanıcı adlarının kaldırılması, durak kelimelerin kaldırılması, hashtag ve url lerin kaldırılması, verilerin sayısallaştırılması(öznitelik çıkarımı ve ağırlıklandırılması) işlemleri uygulanmıştır.
İstenmeyen karakterler Regex kullanılarak kaldırılmıştır. Durak kelimelerin kaldırılmasında ise, öncelikle durak kelimelerin olduğu bir txt dosyası hazırlanmıştır. Sonrasında bu dosya okunup, eşleşen kelimeler döngü ve if blokları yardımıyla kelimeler kaldırılmıştır. 

Verilerin Sayısallaştırılması:
Sayısallaştırma aşamasından önce işaretlediğimiz verilerin(730 adet) eğitim ve test verisi olarak ayrılması gerekmektedir. train_test_split yardımıyla veriler ayrılmıştır. %75 lik kısmı eğitim, %25 lik kısmı test için kullanılmıştır. 
Metin sınıflandırma işleminde sıkça kullanılan Tf-Idf terim ağırlıklandırma ve N-Gram öznitelik elde etme yöntemleri kullanılarak veriler sayısallaştırılmıştır.  

Bu çalışmada ilk olarak Tf-Idf vektörleyicisi belgelerin %90' dan fazlasında ve 5' ten azında görünen terimleri göz ardı edilerek eğitim verilerine uydurulmuştur(TfidfVectorizer).

Lojistik Regresyon Sınıflandırması:
Bu çalışmada makine öğrenmesi sınıflandırma algoritmalarından Lojistik Regresyon kullanılmıştır. fit fonksiyonu ile model eğitilmiştir. Sonrasında predict fonksiyonu ile eğitilen model test edilmiştir. 
Toplamda yapılan doğru tahmin oranını(başarı skorunu) öğrenebilmek için Accuracy Score kullanılmıştır. Bu yöntem uygulandığında başarı skoru 0.7213 gelmiştir. Bu değeri detaylı incelemek için Confisuon Matrix yapısından yararlanılmıştır. Bu yapı bize modelin kaç tane doğru ve yanlış tahmin sayısını göstermektedir. 

Bu işlemlerden sonra işaretlenmemiş veriler modele verilerek, model tahmin etmiştir. 21397 veriden tahmin sonucunda 20773 negatif, 624 pozitif veri elde edilmiştir. Atılan tweetlerde aynı süreçteki hasta, ölüm ve iyileşen sayıları karşılaştırılmıştır.  

