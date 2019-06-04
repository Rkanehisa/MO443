import cv2
import matplotlib.pyplot as plt
import numpy as np




def match(imgA,imgB,func):
    
    kp1, des1 = func.detectAndCompute(imgA,None)
    kp2, des2 = func.detectAndCompute(imgB,None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    #matches = sorted(matches, key = lambda x:x.distance)

    good = []
    good_without_list = []

    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
            good_without_list.append(m)
    imMatches = cv2.drawMatchesKnn(imgA,kp1,imgB,kp2,matches,None,flags=2)
    #dst = cv2.warpPerspective(imgA,H,(imgB.shape[1] + imgB.shape[1], imgB.shape[0]))
    dst = None
    
    return imMatches,dst

imgA = cv2.imread("img/foto1A.jpg")
imgB = cv2.imread("img/foto1B.jpg")

sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create()
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
orb = cv2.ORB_create()
imMatches,dst =  match(imgA,imgB,orb)
#cv2.imwrite("matches.jpg", imMatches)
plt.imshow(imMatches)
plt.show()
cv2.waitKey()
cv2.destroyAllWindows()

"""
cv2.imwrite("output.jpg",dst)
plt.imshow(dst)
plt.show()
cv2.
"""