import cv2
import imutils
#from constantes import *
carro_esquerda = 0
carro_direita = 0
largura_linha = 380

largura_min = 80  # Largura minima do retangulo
altura_min = 80  # Altura minima do retangulo
offset = 6  # Erro permitido entre pixel
pos_linha_esquerda = 400 #550  # Posição da linha de contagem
pos_linha_direita = 400
pos_linha = 400
delay = 60  # FPS do vídeo
detec = []

def mouse(event, x, y, flags, param):
    print('=====================')
    print('event:', event)
    print('x: ',x)
    print('y: ',y)
    print('flags: ',flags)
cv2.namedWindow('Video Original')
cv2.setMouseCallback('Video Original', mouse)

def pega_centro(x, y, largura, altura): # Gera um ponto no centro do retangulo
    """
    :param x: x do objeto
    :param y: y do objeto
    :param largura: largura do objeto
    :param altura: altura do objeto
    :return: tupla que contém as coordenadas do centro de um objeto
    """
    x1 = largura // 2
    y1 = altura // 2
    cx = x + x1
    cy = y + y1
    return cx, cy

def set_info(detec): # Detecta os carros que passarem pela linha
    global carro_direita
    global carro_esquerda

    for (x, y) in detec: # Detecta carros a direita
        if (pos_linha_direita + offset) > y > (pos_linha_direita - offset) and x < largura_linha:
            carro_direita += 1
            cv2.line(frame1, (25, pos_linha_direita), (largura_linha, pos_linha_direita), (0, 127, 255), 5)
            detec.remove((x, y))
            print("Carros detectados até o momento: " + str(carro_direita))
            print("x, y: ", x, y)

        
        #elif (pos_linha + offset) > y > (pos_linha - offset) and x > largura_linha:
        if x > 464 and x < 815 and y > 394 and y < 408:
            print('ESSE AQUI: ', (pos_linha_esquerda + offset) ,y, (pos_linha_esquerda - offset) ,x, largura_linha)
            carro_esquerda += 1
            cv2.line(frame1, (25+435, pos_linha), (largura_linha+435, pos_linha), (0, 255, 0), 5) 
            detec.remove((x, y))
            print("Carros detectados até o momento:: " + str(carro_esquerda))
            print("x, y: ", x, y)


def show_info(frame1, dilatada): # Mostra o texto dos carros 
    text = f'Carros: {carro_direita}' 
    text2 = f'Carros: {carro_esquerda}'
    cv2.putText(frame1, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 100, 200), 5)
    cv2.putText(frame1, text2, (550, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (250, 0, 190), 5)
    cv2.imshow("Video Original", frame1)
    #cv2.imshow("Detectar", dilatada)


carros = caminhoes = 0
cap = cv2.VideoCapture('C:/Users/guilh/Desktop/Deep Learning/Open CV/Projetos/contadorDeVeiculos/videos/video1.mp4')
subtracao = cv2.bgsegm.createBackgroundSubtractorMOG()  # Pega o fundo e subtrai do que está se movendo

while True:
    ret, frame1 = cap.read()  # Pega cada frame do vídeo
    tempo = float(1 / delay)
    sleep(tempo)  # Dá um delay entre cada processamento
    frame1 = imutils.resize(frame1, width=850) # Redimenciona video
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)  # Pega o frame e transforma para preto e branco
    blur = cv2.GaussianBlur(grey, (3, 3), 5)  # Faz um blur para tentar remover as imperfeições da imagem
    img_sub = subtracao.apply(blur)  # Faz a subtração da imagem aplicada no blur
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))  # "Engrossa" o que sobrou da subtração
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (
        5, 5))  # Cria uma matriz 5x5, em que o formato da matriz entre 0 e 1 forma uma elipse dentro
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)  # Tenta preencher todos os "buracos" da imagem
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)

    contorno, img = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.line(frame1, (25, pos_linha), (largura_linha, pos_linha), (255, 127, 0), 3) # Mostra linha direita
    cv2.line(frame1, (25+435, pos_linha), (largura_linha+435, pos_linha), (0, 0, 255), 3) # Mostra linha esquerda
    
    for (i, c) in enumerate(contorno):
        (x, y, w, h) = cv2.boundingRect(c)
        validar_contorno = (w >= largura_min) and (h >= altura_min) # Validador
        if not validar_contorno:
            continue

        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        centro = pega_centro(x, y, w, h)
        detec.append(centro)
        cv2.circle(frame1, centro, 4, (0, 0, 255), -1)

    set_info(detec)
    show_info(frame1, dilatada)

    if cv2.waitKey(10) == 27:
        break

cv2.destroyAllWindows()
cap.release()
