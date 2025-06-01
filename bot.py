from telegram.ext import Updater, MessageHandler, Filters
from telegram import Bot
from deepface import DeepFace
import joblib
import uuid
import os

knn = joblib.load("reconhecimento_facial.pkl")  # Arquivo que você exportou do Colab
threshold = 20
def prever_imagem(img_path):
    try:
        # Extrai o embedding
        embedding = DeepFace.represent(img_path=img_path, model_name="Facenet512", enforce_detection=False)[0]["embedding"]

        # Verifica distância ao vizinho mais próximo
        distances, _ = knn.kneighbors([embedding], n_neighbors=1)
        distancia = distances[0][0]
        pred = knn.predict([embedding])[0]

        print(f"Distância da imagem: {distancia:.4f} | Threshold: {threshold:.4f}")

        if distancia > threshold:
            return "Identidade desconhecida"
        else:
            return pred
    except Exception as e:
        return f"Erro: {e}"



def handle_photo(update, context):
    file = update.message.photo[-1].get_file()
    img_name = f"{uuid.uuid4()}.jpg"
    file.download(img_name)

    resultado = prever_imagem(img_name)
    update.message.reply_text(f"Resultado: {resultado}")

    os.remove(img_name)

def main():
    TOKEN = "8152620137:AAGGnBDTnU1KUGXbEOI-SZP8IkJl5MrDniY" 
    updater = Updater(token=TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(MessageHandler(Filters.photo, handle_photo))

    updater.start_polling()
    print("Bot está rodando...")
    updater.idle()

if __name__ == '__main__':
    main()
