# 🧠 PyTorch_Bölüm-7.1 — Özel AI API Geliştirme

## 🎯 Amaç

Bu çalışma, **PyTorch_Bölüm-7 – Model Servisleştirme ve API** aşamasının devamıdır.  
Bu bölümde hazır modeller yerine, **tamamen kendi geliştirdiğim yapay zekâ servisleri** tasarlanmış ve gerçek çalışma ortamına entegre edilmiştir.  

Amaç, PyTorch tabanlı özel modellerin üretim düzeyinde servisleştirilmesi ve bu servislerin web tabanlı arayüzlerle etkileşime sokulmasıdır.  
Her proje, bağımsız yapıda çalışan özgün bir **API servisi** olarak yapılandırılmıştır.

---

## 🧩 Servisler

### 1️⃣ Özel LLM Servisi  
- **Amaç:** Kendi dil modelimin (LLM) API olarak sunulması.  
- **Model Tabanı:** Transformer tabanlı encoder–decoder mimarisi  
- **Teknolojiler:** `PyTorch`, `JavaScript`  
- **Özellik:**  
  Sıfırdan geliştirilmiş model; **DropPath**, **MultiHeadAttention**, **FeedForward**, **Positional Encoding**, **Token Embedding**  yapılarıyla oluşturulmuştur.  
  Model, çok katmanlı encoder–decoder mimarisiyle **metin üretimi**, **özetleme** ve **anlam çıkarımı** yapabilir.  
- **Çalışma Prensibi:**  
  Kullanıcı girdisine göre dinamik yanıtlar üretir ve bu yanıtlar API aracılığıyla web arayüzünde gösterilir.  
- **Ek Özellik:**  
  Bu modelde, modern Transformer yapısına ek olarak **socket kütüphanesi anlatımı** ve LLM üzerinde **socket tabanlı etkileşim özellikleri** de uygulanmıştır.  

#### 🔹 Arayüz Görseli
<img src="MyAPI/- Kendi API'mizi Kullanalım -/Torch - LLM -/Torch-LLM.png" width="750"/>

---

### 2️⃣ Hyso CNN Servisi
- **Amaç:** Kendi geliştirdiğim **Hyso** kütüphanesiyle oluşturulan CNN modellerini web üzerinden çalıştırmak.  
- **Teknolojiler:** `PyTorch`, `HTML`, `CSS`, `JavaScript`  
- **Çalışma Prensibi:**  
  Görsel yüklenir → model tahmini gerçekleştirir → sonuç anlık olarak ekranda gösterilir.  
- **Özellik:**  
  Eğitim, tahmin ve arayüz etkileşimi tamamen sıfırdan kodlanmıştır.  

#### 🔹 Arayüz Görseli
<img src="MyAPI/- Kendi API'mizi Kullanalım -/Torch - CNN -/Torch - CNN.png" width="750"/>

---

### 3️⃣ ML Regresyon Servisi
- **Amaç:** Basit makine öğrenmesi modellerini (örneğin Linear veya Polynomial Regression) API olarak servisleştirmek.  
- **Model Tabanı:** `scikit-learn` ve `PyTorch` ile eğitilmiş regresyon modelleri  
- **Teknolojiler:** `Python`, `FastAPI`, `HTML`, `JavaScript`  
- **Özellik:**  
  Model, kullanıcıdan aldığı giriş verilerini kullanarak tahmin işlemini gerçekleştirir ve sonucu JSON formatında döner.  
  Eğitim süreci sonrası model `.pkl` veya `.pt` dosyası olarak kaydedilmiş, API’ye entegre edilmiştir.  
- **Çalışma Prensibi:**  
  Web arayüzü üzerinden girilen sayısal veriler API’ye gönderilir → model tahmin yapar → sonuç tarayıcıda görüntülenir.  

#### 🔹 Arayüz Görseli
<img src="MyAPI/- Kendi API'mizi Kullanalım -/Sklearn - ML -/ML.png" width="750"/>

---

## 💡 Ek Bilgi

Bu çalışma, yalnızca model geliştirmeye değil;  
**servis mimarisi**, **entegrasyon**, ve **gerçek zamanlı etkileşim** aşamalarına odaklanır.  
Tüm kodlar sıfırdan oluşturulmuş, herhangi bir hazır servis veya framework’ten türetilmemiştir.  

---

> 🔹 Bu proje, **PyTorch Eğitim Serisi’nin 7.1. bölümü** olup,  
> yapay zekâ modellerinin sıfırdan geliştirilen **özgün API servislerine dönüştürülmesi** sürecini temsil eder.  
> 🔹 Amaç, kendi modellerini gerçek dünyada çalışabilir hale getirmek ve bu süreci uçtan uca yönetebilmektir.  
