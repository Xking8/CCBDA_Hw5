import pandas as pd
import numpy as np
from scipy.stats.stats import pearsonr
from scipy.spatial.distance import cosine
#a = [[1,2,3],[4,5,6],[7,8,9]]
user_num = 6040
movie_num = 3952
w, h = movie_num+1, user_num+1;
Matrix = [[0 for x in range(w)] for y in range(h)] 

#df.to_csv('mtraining.dat')
df = pd.read_csv('training.dat')
for i in range(747907):
  #print df.get_value(i,0,True), df.get_value(i,1,True)  
  Matrix[   df.get_value(i,0,True)  ][ df.get_value(i,1,True) ] = df.get_value(i,2,True)
#print df
#print Matrix
#print Matrix[0]
#print Matrix[1]
#print pearsonr(Matrix[0],Matrix[1])
Usr_rating_mean = [0 for x in range(user_num)]
for i in xrange(1,user_num):
  rating_sum = sum(Matrix[i])
  sum_count = np.count_nonzero(Matrix[i])
  Usr_rating_mean[i] = rating_sum / sum_count

"""

error = 0
error_count = 1
for i in xrange(1,user_num):
  for j in xrange(1,movie_num):
   if Matrix[i][j] is not 0:
    predict = 0
    simsum = 0
    simcount =0
    for k in xrange(i+1,user_num):
      if np.count_nonzero(Matrix[i]) and np.count_nonzero(Matrix[k]):
        sim = cosine(Matrix[i],Matrix[k])
        predict += sim * (Matrix[k][j] ) #- Usr_rating_mean[k])
        simsum += sim
        simcount += 1
        
    simmean = simsum / simcount
    predict = Usr_rating_mean[i] + predict / simcount
    #if Matrix[i][j] is not 0:
    error += abs(Matrix[i][j]-predict)
    error_count += 1

    print "(",i,",",j,")=",Matrix[i][j],",",simmean," predict: ",predict," for now error= ",error/error_count
"""

dftest = pd.read_csv('testing.dat')
fr = open("testing2.dat")
fw = open("result.dat","w")
fw.write("uID-movieID-timestamp,rating\n")
for line in fr.read().splitlines():
  UID,MID,time = line.split(",")
  if int(UID) < 5:
      iUID = int(UID)
      iMID = int(MID)
      predict = 0
      simsum = 0
      simcount = 0
      for k in range(user_num+1):
        if np.count_nonzero(Matrix[iUID]) and np.count_nonzero(Matrix[k]):
          sim = cosine(Matrix[iUID],Matrix[k])
          predict += sim * Matrix[k][iMID]
          simcount += 1
      predict = Usr_rating_mean[iUID] + predict/simcount
      wstr = UID + "-" + MID + "-" + time + "," + str(predict) + "\n"
  else:
      wstr = UID + "-" + MID + "-" + time +",3.785\n"
  print wstr
  fw.write(wstr)


















