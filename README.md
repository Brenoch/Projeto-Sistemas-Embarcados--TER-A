# Projeto: Detector de Capacete com YOLOv8

Este projeto utiliza o modelo YOLOv8 da Ultralytics para detectar capacetes, bonés/chapeús e pessoas sem proteção na cabeça em imagens ou vídeo. O objetivo é promover segurança em ambientes industriais ou de construção.

## Descrição do Hardware Utilizado
- Computador com Windows 10 ou superior (recomendado)
- Webcam USB ou embutida (para detecção em tempo real)
- Requisitos mínimos: 4GB RAM, processador dual-core
- Para treinamento acelerado, recomenda-se GPU NVIDIA compatível com CUDA

## Descrição do Software Utilizado
- Sistema operacional: Windows 10/11
- Python 3.8 ou superior
- Bibliotecas: Ultralytics (YOLOv8), OpenCV
- Ferramentas de anotação: Roboflow (online) ou CVAT (offline)
- Editor de código: Visual Studio Code

## Modelo de Inteligência Computacional
O projeto utiliza o modelo YOLOv8 (You Only Look Once, versão 8) da Ultralytics, especializado em detecção de objetos em tempo real.
- O modelo foi treinado para três classes: `helmet`, `no_helmet` e `hat`.
- O treinamento foi realizado com imagens anotadas e customização do arquivo `data.yaml` para refletir as classes do projeto.
- Foram realizados ajustes no script de detecção para exibir alertas visuais e mensagens específicas para cada classe detectada.
- O modelo padrão da Ultralytics foi adaptado para o contexto de segurança industrial, com esforço de desenvolvimento para organização do dataset, anotação personalizada e integração com webcam.

## Estrutura do Projeto
```
helmet_dataset/
  data.yaml           # Configuração do dataset
  train/
    images/           # Imagens para treinamento
    labels/           # Anotações YOLO para treinamento
  valid/
    images/           # Imagens para validação
    labels/           # Anotações YOLO para validação
models/
  helmetV5_1.pt       # Modelo YOLOv5 (exemplo)
  hemletYoloV8_100epochs.pt # Modelo YOLOv8 treinado
  hemletYoloV8_25epochs.pt  # Modelo YOLOv8 treinado
sistema capacete.py   # Script de detecção
```

## Passo a Passo para Treinamento

1. **Preparação do Dataset**
   - Colete imagens de pessoas com capacete, sem capacete e com boné/chapéu.
   - Anote as imagens usando Roboflow ou CVAT, exportando em formato YOLOv5 TXT.
   - Distribua 80% das imagens/anotações em `train/` e 20% em `valid/`.

2. **Configuração do arquivo `data.yaml`**
   - Exemplo:
     ```yaml
     train: ./helmet_dataset/train/images
     val: ./helmet_dataset/valid/images
     nc: 3
     names: ['hat', 'helmet', 'no_helmet']
     ```

3. **Instalação das Dependências**
   - Instale a biblioteca Ultralytics:
     ```powershell
     pip install -U ultralytics
     ```

4. **Treinamento do Modelo**
   - Execute no terminal:
     ```powershell
     yolo task=detect mode=train model=yolov8n.pt data=helmet_dataset/data.yaml epochs=50 imgsz=640
     ```
   - Ou crie um script `train.py`:
     ```python
     from ultralytics import YOLO
     model = YOLO('yolov8n.pt')
     results = model.train(
         data='helmet_dataset/data.yaml',
         epochs=50,
         imgsz=640,
         project='training_results',
         name='helmet_run1'
     )
     ```

5. **Utilização do Modelo Treinado**
   - Após o treinamento, copie o arquivo `best.pt` para a pasta `models/`.
   - No script de detecção, altere para:
     ```python
     model = YOLO('models/best.pt')
     ```

## Execução do Script de Detecção
- Execute o script `sistema capacete.py` para detectar capacetes em tempo real usando a webcam.

## Requisitos
- Python 3.8+
- Ultralytics (YOLOv8)
- OpenCV

## Observações
- Certifique-se de que as classes em `data.yaml` estejam na mesma ordem das anotações.
- O modelo pode ser treinado com diferentes quantidades de épocas e tamanhos de imagem conforme necessidade.