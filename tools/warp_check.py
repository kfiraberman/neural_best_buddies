import cv2

awarped= cv2.imread("/Users/gunnarenserro/Documents/github/shark-update/code/neural_best_buddies/results/lion_cat/warp_AtoM.png")
bwarped= cv2.imread("/Users/gunnarenserro/Documents/github/shark-update/code/neural_best_buddies/results/lion_cat/warp_BtoM.png")


cv2.imshow("warping",cv2.addWeighted(awarped,0.5,bwarped,0.5,0))
cv2.waitKey(0)