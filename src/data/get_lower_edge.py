import cv2
import numpy as np 

#Example array
edged = np.zeros((24,24))
edged[0,0] = 1
edged[5,1] = 1
edged[10,4] = 1
edged[15,9] = 1
edged[20,16] = 1

#Find the lowest positive value in every column
#In order to compare points by their y coordinate, flip all the points later on
edgelist = []
for i in range(len(edged)):
    for j in range(len(edged[i])-1,-1,-1):
        if edged[i,j] != 0:
            edgelist.append((j,i))
            break

#Divide in half and take minimum of both sides; find "vertex"; flip all the points back
new_list = []
vertex = (5000,5000)
for x in range(0,len(edgelist)//2+1):
    min_point = min(edgelist[x],edgelist[len(edgelist)-x-1])
    new_list.append(min_point)
    if min_point[0] < vertex[0]:
        vertex = min_point
vertex = vertex[::-1]
edgelist = []
for point in new_list:
    edgelist.append(point[::-1])

#Iterate through list of points and find values of parabola that are lowest
a_min = 500
for point in edgelist:
    if (point != vertex):
        a_min = min(a_min,(point[1] - vertex[1])/((point[0]-vertex[0])**2))

#For all points past the parabola, make them 0s
mask = np.ones((len(edged),len(edged[0])))
for i in range(len(mask)):
    for j in range(max(int((a_min*(i-vertex[0])**2+vertex[1])),0),len(mask[i])):
        mask[i,j] = 0

print(mask)











