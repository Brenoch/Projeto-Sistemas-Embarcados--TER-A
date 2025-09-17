from ultralytics import YOLO
import cv2
import os


# Carregue o modelo YOLOv8 treinado para capacete (.pt)
model = YOLO('models/hemletYoloV8_25epochs.pt')


# Inicialize a captura de vídeo da câmera padrão
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Redimensiona o frame para acelerar o processamento
    # Mantém a proporção original para não distorcer a imagem
    orig_h, orig_w = frame.shape[:2]
    target_w = 640
    scale = target_w / orig_w
    target_h = int(orig_h * scale)
    resized_frame = cv2.resize(frame, (target_w, target_h))


    # Executa a detecção no frame redimensionado
    results = model(resized_frame)[0]

    # Itera sobre as caixas de detecção retornadas
    for box in results.boxes:
        # Coordenadas do bounding box (precisam ser reescaladas para o frame original)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1, y1, x2, y2 = int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale)
        conf = box.conf[0].item()  # confiança da detecção
        cls = int(box.cls[0])      # índice da classe detectada
        label = model.names[cls]  # nome da classe (helmet/no_helmet)

        # Define a cor do retângulo (verde para capacete, vermelho para sem capacete)
        color = (0, 255, 0) if 'helmet' in label.lower() else (0, 0, 255)

        # Desenha o retângulo e o texto de alerta na imagem
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.2f}"
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Exibe o frame com as detecções
    cv2.imshow('Helmet Detection (YOLOv8)', frame)

    # Sai do loop ao pressionar a tecla ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Libera a câmera e fecha todas as janelas
cap.release()
cv2.destroyAllWindows()
