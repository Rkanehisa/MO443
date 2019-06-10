import cv2
import matplotlib.pyplot as plt
import numpy as np


#Crop na imagem de saída
def trim(frame):
    #crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop top
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    #crop top
    if not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    #crop top
    if not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame

def match(imgA,imgB,func):
    #Converte pra greyscale
    img_ = cv2.cvtColor(imgA,cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(imgB,cv2.COLOR_BGR2GRAY)

    #Computa os keypoints
    kp1, des1 = func.detectAndCompute(img_,None)
    kp2, des2 = func.detectAndCompute(img,None)

    #Calcula a semelhança
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    #Elimila os keypoinsta ruins
    good = []
    good_without_list = []

    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append([m])
            good_without_list.append(m)


    #Desenha os keypoints encontrados
    matches = sorted(matches, key = lambda x:x[0].distance)
    good_matches = matches[:6]
    imMatches = cv2.drawMatchesKnn(imgA,kp1,imgB,kp2,good_matches,None,matchColor=(0,0,255),flags=2)

    #Caso haja mais de 4 pontos
    match = np.asarray(good)
    if len(match[:,0]) >= 4:
        #Aplica a transformação
        src = np.float32([ kp1[m.queryIdx].pt for m in match[:,0] ]).reshape(-1,1,2)
        dst = np.float32([ kp2[m.trainIdx].pt for m in match[:,0] ]).reshape(-1,1,2)

        #Calcula a matrix de homografia
        H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        print(H)
        print()

        #Gera a imagem panorâmica
        dst = cv2.warpPerspective(imgA,H,(img.shape[1] + imgA.shape[1], img.shape[0]))
        dst[0:imgB.shape[0],0:imgB.shape[1]] = imgB
        return imMatches,trim(dst)
    else:
        print("Not enought points")
        return

#Cria os descritores
sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create()
orb = cv2.ORB_create()

imgA = cv2.imread("img/foto5A.jpg")
imgB = cv2.imread("img/foto5B.jpg")


#Utiliza cada um dos descritores para gerar as imagem
imMatches,dst =  match(imgA,imgB,orb)
cv2.imwrite("matches_orb.jpg", imMatches)
cv2.imwrite("output5_orb.jpg",dst)

imMatches,dst =  match(imgA,imgB,sift)
cv2.imwrite("matches_sift.jpg", imMatches)
cv2.imwrite("output_sift.jpg",dst)

imMatches,dst =  match(imgA,imgB,surf)
cv2.imwrite("matches_surf.jpg", imMatches)
cv2.imwrite("output_surf.jpg",dst)