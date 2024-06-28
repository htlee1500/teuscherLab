import torch


# Given a destination file name and a 2D tensor (batch_size) x (sample_length), draw it in a file
def draw(dest, data):
    f = open(dest, "w")

    list_data = data.tolist()

    print(list_data)
    
    for i in range(len(list_data)):
        for x in range(28):
            row = ""
            for y in range(28):
                if list_data[i][x+y] == 1:
                    row += "X"
                    print("got the spike @" + dest + "("+str(i)+","+str(x)+","+str(y)+")")
                else:
                    row += " "
                #val = round(list_data[i][x+y]*10)/10
                #val = list_data[i][x+y]
                #print(val)
                row += str(val) + " "
            f.write(row)
            f.write("\n")

        f.write("\n")
    
def search(data):
    list_data = data.tolist()
    for i in range(len(list_data)):
        for j in range(len(list_data[i])):
            val = list_data[i][j]
            if val != 0:
                print("Found " + str(val) + " in Image " + str(i) + " pixel number " + str(j))
