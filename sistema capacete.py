from ultralytics import YOLO
import cv2

# Carregue o SEU NOVO modelo YOLOv8 treinado com as 3 classes
# Substitua pelo caminho do seu arquivo .pt recém-treinado
try:
    model = YOLO('models/hemletYoloV8_100epochs.pt') # Ex: 'runs/detect/train/weights/best.pt'
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    print("Verifique se o caminho para o arquivo .pt está correto.")
    exit()

# Inicialize a captura de vídeo da câmera padrão
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro: Não foi possível acessar a câmera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Falha ao capturar o frame. Fim do vídeo?")
        break

    # Executa a detecção no frame. Usar 'stream=True' é mais eficiente para vídeos.
    results = model(frame, stream=True, verbose=False) # verbose=False para limpar o console

    # Itera sobre as detecções encontradas
    for r in results:
        for box in r.boxes:
            # Coordenadas da caixa delimitadora
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Confiança da detecção
            conf = box.conf[0].item()

            # Classe detectada
            cls = int(box.cls[0])
            label = model.names[cls]

            # --- LÓGICA DE CORES E ALERTAS BASEADA NA CLASSE ---
            color = (0, 0, 0)  # Padrão: Preto
            alert_text = label

            if label == 'helmet':
                color = (0, 255, 0)  # Verde: Seguro
                alert_text = "Capacete Detectado"
            elif label == 'no_helmet':
                color = (0, 0, 255)  # Vermelho: Perigo
                alert_text = "ALERTA: SEM CAPACETE"
            elif label == 'hat':
                color = (0, 255, 255) # Amarelo: Informativo
                alert_text = "Bone ou Chapeu" 

            # Desenha o retângulo ao redor do objeto detectado
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Prepara e desenha o texto acima do retângulo
            text = f"{alert_text} {conf:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Exibe o frame com as detecções
    cv2.imshow('Detector de Capacete - YOLOv8', frame)

    # Sai do loop ao pressionar a tecla 'ESC' (código 27)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Libera os recursos ao finalizar
cap.release()
cv2.destroyAllWindows()