HandCursor

Controle o cursor do mouse usando gestos da mão, através da webcam, com visão computacional em tempo real.
Este projeto utiliza MediaPipe, OpenCV e PyAutoGUI para detectar landmarks da mão e traduzir movimentos e gestos em ações do mouse, como mover, clicar, arrastar e scrollar.

Projeto de estudo focado em programação em Python, consumo e integração de APIs, visão computacional, interação humano-computador (HCI) e processamento em tempo real.

FUNCIONALIDADES
- Mover o cursor com a mão
- Clique simples com gesto do polegar + base do indicador
- Clique e segurar (drag) com gesto de pinça (polegar + ponta do indicador)
- Scroll com gesto de movimento (com ponta do indicador + ponta do médio)
- Visualização em tempo real dos landmarks
- Movimento suavizado (anti-tremor)
- Otimizações para melhor desempenho (frame skip e resolução reduzida)

GESTOS UTILIZADOS (Gesto / Ação)
- Mão aberta (movendo) / Move o cursor
- Polegar + base do indicador juntos / Clique simples
- Polegar + ponta do indicador juntos / Clique e segurar (drag)
- Afastar polegar da ponta do indicador / Soltar drag


COMO FUNCIONA
- A webcam captura os frames em tempo real
- O MediaPipe detecta 21 landmarks da mão
- Um ponto de referência (base do dedo médio) é usado para mapear o movimento
- As coordenadas da câmera são interpoladas para a resolução da tela
- Gestos são detectados através da distância entre landmarks específicos
- O PyAutoGUI executa as ações do mouse no sistema operacional


TECNOLOGIAS UTILIZADAS
- Python 3.10
- OpenCV
- MediaPipe (HandLandmarker Task)
- NumPy
- PyAutoGUI


REQUISITOS
- Python 3.10
- Webcam funcional
- Sistema operacional:
    Windows
    Linux
    macOS


INSTALAÇÃO
- Clone o repositório
- Instale as dependências:
    pip install opencv-python mediapipe pyautogui numpy
- Baixe o modelo do MediaPipe Hand Landmarker e coloque na raiz do projeto:
    hand_landmarker.task
    (O modelo pode ser obtido na documentação oficial do MediaPipe.)


EXECUTANDO
- No cmd: python handcursor.py
- Pressione ESC para encerrar o programa
- Mantenha a mão dentro do retângulo exibido na tela


DETALHES TÉCNICOS
- O cursor é controlado usando a base do dedo médio, o que reduz tremores
- Suavização EMA aplicada para evitar movimentos bruscos
- Frame skip configurável para melhorar desempenho
- Controle para evitar múltiplos cliques indesejados, com sleep()
- Margens configuradas para limitar a área ativa da câmera


LICENÇA
- Este projeto é apenas para fins educacionais e de estudo.
Sinta-se à vontade para usar, modificar e aprender com ele.
