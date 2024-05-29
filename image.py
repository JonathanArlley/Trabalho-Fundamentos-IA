import cv2
import os

# Caminho da imagem
image_path = r"C:/Users/arlle/OneDrive/Área de Trabalho/Projeto-de-processamento-de-imagem--main (1)/Projeto-de-processamento-de-imagem--main/images.jpg"

# Verificar se a imagem existe no caminho especificado
if not os.path.isfile(image_path):
    print(f"Imagem não encontrada no caminho: {image_path}")
else:
    imageIn = cv2.imread(image_path)

    # Verificar se a imagem foi carregada corretamente
    if imageIn is None:
        print(f"Erro ao carregar a imagem no caminho: {image_path}")
    else:
        grey = cv2.cvtColor(imageIn, cv2.COLOR_BGR2GRAY)

        # Caminho do arquivo XML
        xml_path = r"C:\Users\arlle\OneDrive\Área de Trabalho\Projeto-de-processamento-de-imagem--main (1)\Projeto-de-processamento-de-imagem--main\haarcascade_frontalface_default.xml"

        # Verificar se o arquivo XML existe
        if not os.path.isfile(xml_path):
            print(f"Arquivo Haar Cascade não encontrado no caminho: {xml_path}")
        else:
            human_head_cascade = cv2.CascadeClassifier(xml_path)

            # Verificar se o arquivo XML foi carregado corretamente
            if human_head_cascade.empty():
                print(f"Erro ao carregar o arquivo Haar Cascade no caminho: {xml_path}")
            else:
                human_head = human_head_cascade.detectMultiScale(grey, 1.2, 5)
                for (x, y, w, h) in human_head:
                    cv2.rectangle(imageIn, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.imshow('img', imageIn)
                cv2.waitKey()
                cv2.destroyAllWindows()
