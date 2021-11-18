from tensorflow.keras.models import load_model
import cv2
import numpy as np

#Carregar Modelo
model = load_model("keras_model.h5")

#Carregar Webcam
cam = cv2.VideoCapture(0)

#Para cada cena...
while True:
    #Armazena imagem em variavel
    ret, img = cam.read()

 
    #Formatação
    roi = cv2.resize(img, (224, 224))
    #Formatação
    roi = np.reshape(roi, [1, 224, 224, 3])


    if np.sum([roi])!= 0: 
            
        #Cast
        roi = (roi.astype('float')/255.0)#
            
        #Prediz se é gato ou cachorro
        result = model.predict([[roi]])

        #Recupera o resultado
        result =  result[0]

        #Se for cachorro...
        if result[0]>=result[1]:
            #Define uma string com o nome, chance e porcentagem
            label= 'Cachorro | CHANCE: ' + str(round(result[0]*100, 2)) + "%"
            #Cor BGR
            color = (0, 255, 13)
           
        #Se for Gato...
        else:
            #Define uma string com o nome, chance e porcentagem
            label= 'Gato | CHANCE: ' + str(round(result[1]*100, 2)) + "%"
            #Cor BGR
            color = (251, 255, 0)
                

        #PRINT     imagem     nome        posição               fonte                Largura fonte      cor
        cv2.putText(img,     label,      (30, 40),       cv2.FONT_HERSHEY_DUPLEX,         .6,          color, 2)

    #Plotar imagem
    cv2.imshow("Detector", img)

    #Aguarda pressionar "q" para encerrar o programa
    key = cv2.waitKey(1)
    if key ==  ord('q'):
        break
    elif key ==  ord('Q'):
        break

#Libera acesso à webcam
cam.release()

#Destroi as janelas criada
cv2.destroyAllWindows()

