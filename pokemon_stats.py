import csv
import math
import random
import numpy as np
import scipy.cluster.hierarchy as scip
import matplotlib.pyplot as plt

# takes in a string with a path to a CSV file formatted as in the link above,
# and returns the first 20 data points (without the Generation
# and Legendary columns but retaining all other columns) in a list of dictionarys
def load_data(filepath):
    res = []
    with open(filepath, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            if (int(row['#']) == 16):
                break
            current = {}
            current['Number'] = int(row['#'])
            current['Name'] = row['Name']
            current['Type 1'] = row['Type 1']
            current['Type 2'] = row['Type 2']
            current['Total'] = int(row['Total'])
            current['HP'] = int(row['HP'])
            current['Attack'] = int(row['Attack'])
            current['Defense'] = int(row['Defense'])
            current['Sp. Atk'] = int(row['Sp. Atk'])
            current['Sp. Def'] = int(row['Sp. Def'])
            current['Speed'] = int(row['Speed'])
            res.append(current)
    csvfile.close()
    return res

#takes in one row from the data loaded from the previous
# function, calculates the corresponding x, y values for
# that Pokemon as specified above, and returns them in a single structure.
def calculate_x_y(stats):
    # Attack + Sp. Atk + Speed
    Attack = stats['Attack']
    spAtk = stats['Sp. Atk']
    Speed = stats['Speed']
    x = Attack + spAtk + Speed
    # Defense + Sp. Def + HP
    Defense = stats['Defense']
    spdef = stats['Sp. Def']
    HP = stats['HP']
    y = Defense + spdef + HP
    xy = (x, y)
    return xy

#performs single linkage hierarchical agglomerative clustering on the
# Pokemon with the (x,y) feature representation, and returns a data
# structure representing the clustering.
def hac(dataset):
    datatemp = dataset.copy() # makes a temp copy of all of the data in the data set to manipulate
    res = []
    num = 0
    # until the desired number of rows is reached
    while (num < len(dataset)-1 ):
        # calls a helper method to get all of the distances adn their cordinates in datatemp
        temp = distlist(datatemp)
        # finds the minimum distance
        to_add = find_min(temp)
        row = []
        # makes the row with the values and distances
        val1 =to_add['val1']
        row.append( datatemp[val1])
        val2 = to_add['val2']
        row.append(datatemp[val2])
        row.append(to_add['dist'])
        # if they are both touples
        if (isinstance(row[0], tuple)) and isinstance(row[1], tuple):
            listtemp = []
            listtemp.append(row[0])
            listtemp.append(row[1])
            row[0] = dataset.index(datatemp[val1])# gets the index from the original set
            row[1] =dataset.index(datatemp[val2])# gets the index from the original set
            # deletes the values from the temp set
            if(val1<val2):
                del datatemp[val1]
                del datatemp[val2-1]
            else:
                del datatemp[val1]
                del datatemp[val2]
            row.append(2)
            # adds the new cluster
            datatemp.append(listtemp)
        # if one is a tuple and one is a list both this elsif block and the next do the same
        # thing just depends on which index is the one that the list or tuple
        elif (isinstance(row[0], list) and isinstance(row[1], tuple)):
            row.append(len(row[0]) + 1)# adds the ammount of elements
            listtemp = row[0].copy() # makes a copy of row[0[
            listtemp.append(row[1])
            tu =(row[0])[len(row[0]) - 1]
            tup = dataset.index(tu)
            row[1] = dataset.index(datatemp[val2])# gets the index from the original set
            # deletes the values from the temp set
            if (val1 < val2):
                del datatemp[val1]
                del datatemp[val2 - 1]
            else:
                del datatemp[val1]
                del datatemp[val2]
            row[0] = len(dataset) + find_row(res,tup,len(dataset)) # row number pus length of data
            # adds cluster
            datatemp.append(listtemp)
        elif (isinstance(row[1], list) and isinstance(row[0], tuple)):
            row.append(1+ len(row[1]))
            listtemp = row[1].copy()
            listtemp.append(row[0])
            tu = (row[1])[len(row[1]) - 1]
            tup = dataset.index(tu)
            row[0] = dataset.index(datatemp[val1])
            if (val1 < val2):
                del datatemp[val1]
                del datatemp[val2 - 1]
            else:
                del datatemp[val1]
                del datatemp[val2]
            row[1] = len(dataset) + find_row(res,tup,len(dataset))
            datatemp.append(listtemp)
        # if they are both a list
        elif(isinstance(row[1], list) and isinstance(row[0], list)):
            row.append(len(row[0])+len(row[1]))
            # makes the new cluster
            listtemp = row[1].copy()
            listtemp = listtemp+ row[0]
            # gets the indexes
            tu = (row[1])[len(row[1]) - 1]
            tupr1 = dataset.index(tu)
            tu = (row[0])[len(row[0]) - 1]
            tupr0 = dataset.index(tu)
            if (val1 < val2):
                del datatemp[val1]
                del datatemp[val2 - 1]
            else:
                del datatemp[val1]
                del datatemp[val2]
            # gets the row + length vals
            row[0] = len(dataset)+find_row(res,tupr0,len(dataset))
            row[1] = len(dataset) + find_row(res,tupr1,len(dataset))
            datatemp.append(listtemp)
        res.append(row)
        num+=1
    return np.array(res)
# makes a dict containing the distance and the values of the tuples where the distance was taken
def distlist(datatemp):
    temp = []
    for i in range(0, len(datatemp) ):
        for j in range(0, len(datatemp) ):
            if (i != j):
                dict = {}
                dict['val1'] = i
                dict['val2'] = j
                dict['dist'] = singlelink(datatemp[i], datatemp[j])
                temp.append(dict)

    return temp
# does euclidian distance
def eucl_dist(a, b):
    dist = 0
    middle = (a[0]-b[0])**2 +(a[1]-b[1])**2
    dist = math.sqrt(middle)
    return dist

# using the minimum distance as the single linkage algorith discribed finds the smallest distance in the clusters
def singlelink(cluster1, cluster2):
    res = float('inf')
    for i in range(len(cluster1)):
        for b in range(len(cluster2)):
            if(isinstance(cluster1,tuple) and isinstance(cluster2,list) ):
                dist = eucl_dist(cluster1, cluster2[b])
            elif(isinstance(cluster2,tuple) and isinstance(cluster1,list) ):

                dist = eucl_dist(cluster1[i], cluster2)
            elif(isinstance(cluster2,list) and isinstance(cluster1,list)):
                dist = eucl_dist(cluster1[i],cluster2[b])
            else:
                dist = eucl_dist(cluster1,cluster2)
            if dist < res:
                res = dist
    return res

# finds the minimum distance and returns that dictionary
def find_min(list):
    if(len(list)>1):
        min = list[0]
        for i in range(1, len(list)):
            if (min['dist'] >= list[i]['dist']):
                min = list[i]
        return min

# finds the row where the cluster was created
def find_row(list, tuple,length):
    row = 0
    for i in range(0, len(list)):
        if (list[i][0] == tuple):
            row = i
            tuple = i +length
        elif (list[i][1] == tuple):
            row = i
            tuple = i + length
    return row

#takes in the number of samples we want to randomly generate, and returns these samples in a single structure
def random_x_y(m):
    r = np.random.uniform(size=m)
    ist = load_data("../../Desktop/HW4Grader_export/Pokemon.csv")
    list1 = []
    for l in ist:
        list1.append(calculate_x_y(l))
    res = []
    for i in range(len(r)):
        index = int(r[i]*20)
        if((list1[index])[0]<360 and (list1[index])[1]<360  ):
            res.append(list1[index])
        else:
            r[i] = int(np.random.uniform(size=1) * 20)
            i-=1
    return res

#performs single linkage hierarchical agglomerative clustering on the Pokemon with the (x,y)
# feature representation, and imshow the clustering process.
def imshow_hac(dataset):
    array = np.array(dataset)
    trans = np.transpose(array)
    x,y = trans
    for a,b in zip(x,y):
        rgb = (random.Random().random(), random.Random().random(), random.Random().random())
        plt.scatter(a,b,color=[rgb])
    # plt.plot(x[0],y[0])
    datatemp = dataset.copy()
    res = []
    num = 0
    while (num < len(dataset) -1):
        plt.pause(.1)
        temp = distlist(datatemp)
        to_add = find_min(temp)
        row = []
        val1 = to_add['val1']
        row.append(datatemp[val1])
        val2 = to_add['val2']
        row.append(datatemp[val2])
        row.append(to_add['dist'])
        if (isinstance(row[0], tuple)) and isinstance(row[1], tuple):
            listtemp = []
            listtemp.append(row[0])
            listtemp.append(row[1])
            if (val1 < val2):
                del datatemp[val1]
                del datatemp[val2 - 1]
            else:
                del datatemp[val1]
                del datatemp[val2]
            row.append(2)
            tempx =(row[0])[0],(row[1])[0]
            xs = np.array(tempx)
            tempy =(row[0])[1],(row[1])[1]
            ys = np.array(tempy)
            plt.plot(xs,ys)
            datatemp.append(listtemp)
        elif (isinstance(row[0], list) and isinstance(row[1], tuple)):
            row.append(len(row[0]) + 1)
            listtemp = row[0].copy()
            listtemp.append(row[1])
            if (val1 < val2):
                del datatemp[val1]
                del datatemp[val2 - 1]
            else:
                del datatemp[val1]
                del datatemp[val2]
            datatemp.append(listtemp)
            cords = shortest(row[0], row[1], row[2])
            tempx =((cords[0])[0],(cords[1])[0])
            xs = np.array(tempx)
            tempy = ((cords[0])[1],(cords[1])[1])
            ys = np.array(tempy)
            plt.plot(xs, ys)
        elif (isinstance(row[1], list) and isinstance(row[0], tuple)):
            row.append(1 + len(row[1]))
            listtemp = row[1].copy()
            listtemp.append(row[0])
            if (val1 < val2):
                del datatemp[val1]
                del datatemp[val2 - 1]
            else:
                del datatemp[val1]
                del datatemp[val2]
            datatemp.append(listtemp)
            cords = shortest(row[0],row[1],row[2])
            tempx = ((cords[0])[0], (cords[1])[0])
            xs = np.array(tempx)
            tempy = ((cords[0])[1], (cords[1])[1])
            ys = np.array(tempy)
            plt.plot(xs, ys)

        elif (isinstance(row[1], list) and isinstance(row[0], list)):
            row.append(len(row[0]) + len(row[1]))
            listtemp = row[1].copy()
            listtemp = listtemp + row[0]
            if (val1 < val2):
                del datatemp[val1]
                del datatemp[val2 - 1]
            else:
                del datatemp[val1]
                del datatemp[val2]
            datatemp.append(listtemp)
            cords = shortest(row[0], row[1], row[2])
            tempx = ((cords[0])[0], (cords[1])[0])
            xs = np.array(tempx)
            tempy = ((cords[0])[1], (cords[1])[1])
            ys = np.array(tempy)
            plt.plot(xs, ys)
        res.append(row)
        num += 1

    plt.pause(.1)
    plt.show(block=True)
    return 0

def shortest(cluster1,cluster2,distance):
    st =[]
    for i in range(len(cluster1)):
        for b in range(len(cluster2)):
            if(isinstance(cluster1,tuple) and isinstance(cluster2,list) ):
                if( eucl_dist(cluster1, cluster2[b]) == distance):
                    st.append(cluster1)
                    st.append(cluster2[b])
            elif(isinstance(cluster2,tuple) and isinstance(cluster1,list) ):
                dist = eucl_dist(cluster1[i], cluster2)
                if (eucl_dist(cluster1[i], cluster2) == distance):
                    st.append(cluster1[i])
                    st.append(cluster2)
            elif(isinstance(cluster2,list) and isinstance(cluster1,list)):
                dist = eucl_dist(cluster1[i],cluster2[b])
                if (eucl_dist(cluster1[i], cluster2[b]) == distance):
                    st.append(cluster1[i])
                    st.append(cluster2[b])
            else:
                dist = eucl_dist(cluster1,cluster2)
                if (eucl_dist(cluster1, cluster2) == distance):
                    st.append(cluster1)
                    st.append(cluster2)

    return st


ist = load_data("../../Desktop/HW4Grader_export/Pokemon.csv")
list1 = []
# print(type(list1))
for l in ist:
    list1.append(calculate_x_y(l))
# print(list1)
ar = np.array(list1)
um = hac(list1)
imshow_hac(list1)