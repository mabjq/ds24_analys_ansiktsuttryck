# Analys av ansiktsuttryck och känslor

## Projektöverblick

Uppgiften var att i ett grupparbete förstå vad tillhandahållen kod gör, få koden att fungera på datorn och i mån av tid utveckla den.

Till uppgiften fanns en förtränad CNN modell i en Keras-fil, en Haar Cascade klassificerare, en skriptfil för analysen och en för Streamlit appen. I en requirements.txt fanns också det som behövdes för installation.

## Filer

- **`Analys_ansikte-känslouttryck.pdf`**:  
  Genomgång av uppgift, kod, felsökning och resultat.

- **`Analyzer.py`**:  
  Kod för att bearbeta videofil, hitta ansikten, prediktera känslouttryck samt skapa utdatavideofil.

- **`Analyzer_orginal.py`**:  
  Orginal Kod för att bearbeta videofil, hitta ansikten och prediktera känslouttryck.

- **`App.py`**:  
  Kod för att skapa Streamlit appen med inställningar för frames to skip och confidence level.

- **`haarcascade_frontalface_default.xml`**:  
  Klassificerings algoritm för att hitta ansikten i gråskala.

- **`Requirements.txt`**:  
  Förutsättningar för att kunna köra koden i Python.

- **`inspect_model.py`**:  
  Kod för att undersöka strukturen på den tillhandahållna CNN modellen.

- **`modelv1.keras`**:  
  CNN modell FER2013. För stor för att laddas upp men finns för nedladdning på nätet.
