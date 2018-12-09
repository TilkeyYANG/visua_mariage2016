# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 14:56:11 2018

@author: TilkeyYang
"""



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Initialising PySpark.SQL
from pyspark import SparkContext; 
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession 
from pyspark.sql.functions import col
# Initialising SparkContext
sc = SparkContext.getOrCreate(); 
sqlContext = SQLContext(sc);

# CD
import os
#os.chdir('D:\\ML\\Visua\\mar_fr_2016') #Abspath on my computer
os.path.realpath('Mar2016_Ver181204.py')
relat_path = os.path.abspath('.') 
cwd = os.getcwd()

# Import DataFrame
data = pd.read_csv("mar2016.csv", sep=';', skiprows = 1,
                   names=[
                         'ANAIS1',  'DEPNAIS1',	'SEXE1',	'INDNAT1',	'ETAMAT1',	
                         'ANAIS2',	'DEPNAIS2', 'SEXE2',	'INDNAT2',	'ETAMAT2',	
                         'AMAR',	'MMAR',	'JSEMAINE',	'DEPMAR',	
                         'DEPDOM',	'TUDOM',	'TUCOM',	'NBENFCOM'])

width = 0.6 # For barplots
plt.style.use('dark_background')

# Autolabel Function
def autolabel_percentage(rects, nb_rects, total):
    i = 0  
    for rect in rects:
      if i < nb_rects:
        height = rect.get_height()
        percentage = height/total*100
        plt.text(rect.get_x()+rect.get_width()/2.-0.2, total/150+height,
                 '   %s%%' % int(percentage), size=8, color='#F6BFBC', 
                 family='sans-serif', style='normal', horizontalalignment='center')
        i+=1

# Auto set frame Function
def set_spines():
  framex = plt.gca()
  framex.spines['top'].set_visible(False)
  framex.spines['right'].set_visible(False)
  framex.spines['left'].set_visible(True)
  framex.spines['left'].set_linewidth(1.2)
  framex.spines['left'].set_linestyle('--')
  framex.spines['bottom'].set_visible(True)
  framex.spines['bottom'].set_linewidth(1.2)
  framex.spines['bottom'].set_linestyle('--')
  plt.grid(True, linestyle = "-", color = "w", linewidth = "0.08")

# Auto set xlim
def auto_set_xlim(xmin = 0, xmax = 100):
  plt.xlim = xmin
  plt.xlim = xmax

# =============================================================================
# BARPLOT
# AE => fig_AgeEcart 
# =============================================================================
figAE = plt.figure()

# Calculate difference of age
age_ecart = np.abs(data.ANAIS1 - data.ANAIS2)
# Counting
count_AE = pd.value_counts(age_ecart.values, sort=False)
# Plotting
age = list(range(len(count_AE)))
barAE = plt.bar(age, count_AE, width=width, color='#a2d7dd', label='FEMME', edgecolor = None, linewidth = 0)
autolabel_percentage(barAE, 16, len(data))

# Select Axis area
plt.xlim(-0.5, 15.5)
plt.grid(False, linestyle = "-", color = "w", linewidth = "0.1")

# Frame format
frame1 = plt.gca()
set_spines()
frame1.spines['left'].set_visible(False)

# Auto-Saving
plt.savefig('./src/figAE.png', 
            format='png', bbox_inches='tight', transparent=True, dpi=240) 
plt.show()



# =============================================================================
# SCATTER
# AS => fig_AgeStatus_Individuel
# =============================================================================
figAS = plt.figure()

# To Seperate Age By Sex
# AS_F => Females' Age List
# AS_M => Males' Age List
AS_F = []; AS_M = [];
for i in range(len(data)):
  if ((data.SEXE1[i] == 'M') or ((data.SEXE1[i] == data.SEXE2[i]))) :
    AS_M.append(2016 - data.ANAIS1[i])
    AS_F.append(2016 - data.ANAIS2[i])
  else :
    AS_F.append(2016 - data.ANAIS1[i])
    AS_M.append(2016 - data.ANAIS2[i])

# Etamat could be treated with the sum(ETAMAT1, ETAMAT2)
sum_ETA = (data.ETAMAT1 + data.ETAMAT2)

# Create a new dataframe with [ Female's Age | Male's Age | Sum of Status ]
df_AS= pd.DataFrame({'AGE_F': AS_F,
                     'AGE_M': AS_M,
                     'SUMETA': sum_ETA})

fig_pts = sns.scatterplot(x="AGE_F", y="AGE_M", hue="SUMETA", data=df_AS, palette="Set2", size = 0.002,
                          alpha=0.9, marker='x', legend=None)

set_spines()
plt.grid(False)

plt.savefig('./src/figAS.png', 
            format='png', bbox_inches='tight', transparent=True, dpi=600) 
plt.show()



# =============================================================================
# PLOT
# AM => fig_AgeMariage_Count
# =============================================================================

# Transform data in to a List "AM" of 465450 rows of individual person:
AM = pd.DataFrame(data[['SEXE1','ANAIS1','ETAMAT1']])
AM.columns=['SEXE', 'ANAIS', 'ETAMAT']
AM2 = pd.DataFrame(data[['SEXE2','ANAIS2','ETAMAT2']])
AM2.columns=['SEXE', 'ANAIS', 'ETAMAT']
AM = AM.append(AM2, ignore_index=True)

# Switching Pandas DF into SPARK SQL DF
sql_AM = sqlContext.createDataFrame(AM).toDF("SEXE", "ANAIS", "ETAMAT")

# Put sql_AM into cache
sql_AM.cache()

# Select Males and Etamat=1, Order by ANAIS
AM_M1 = sql_AM.filter("SEXE LIKE 'M'").filter("ETAMAT == 1").groupBy("ANAIS").count()
# Changing SQL Dataframe back to Pandas DataFrame
AM_M1 = AM_M1.orderBy("ANAIS").toPandas()
# select Males and Etamat=2, Order by ANAIS
AM_M2 = sql_AM.filter("SEXE LIKE 'M'").filter("ETAMAT == 3").groupBy("ANAIS").count()
AM_M2 = AM_M2.orderBy("ANAIS").toPandas()
# select Males and Etamat=3, Order by ANAIS
AM_M3 = sql_AM.filter("SEXE LIKE 'M'").filter("ETAMAT == 4").groupBy("ANAIS").count()
AM_M3 = AM_M3.orderBy("ANAIS").toPandas()
# Same for the Females
AM_F1 = sql_AM.filter("SEXE LIKE 'F'").filter("ETAMAT == 1").groupBy("ANAIS").count()
AM_F1 = AM_F1.orderBy("ANAIS").toPandas()
AM_F2 = sql_AM.filter("SEXE LIKE 'F'").filter("ETAMAT == 3").groupBy("ANAIS").count()
AM_F2 = AM_F2.orderBy("ANAIS").toPandas()
AM_F3 = sql_AM.filter("SEXE LIKE 'F'").filter("ETAMAT == 4").groupBy("ANAIS").count()
AM_F3 = AM_F3.orderBy("ANAIS").toPandas()
  
# Release sql_AM from cache 
sql_AM.unpersist()

# Calculating Age
list_AM = [AM_M1, AM_M2, AM_M3, AM_F1, AM_F2, AM_F3]
for am in list_AM:
  am.ANAIS = 2016-am.ANAIS
  am.columns=['AGE', 'COUNT']
  
# Ploting
figAM1 = plt.figure()
set_spines()
plt.plot(AM_M1.AGE, AM_M1.COUNT, color = '#8EE5EE', marker = '*')
plt.plot(AM_F1.AGE, AM_F1.COUNT, color = '#FF4500', marker = '*')
auto_set_xlim(0,100)         
plt.savefig('./src/figAM1.png', format='png', bbox_inches='tight', 
            transparent=True, dpi=300) 
plt.show()

figAM2 = plt.figure()         
set_spines()
plt.plot(AM_M2.AGE, AM_M2.COUNT, color = '#53868B', marker = '*')
plt.plot(AM_F2.AGE, AM_F2.COUNT, color = '#CD8162', marker = '*')
auto_set_xlim(0,100) 
plt.savefig('./src/figAM2.png', format='png', bbox_inches='tight', 
            transparent=True, dpi=300) 
plt.show()

figAM3 = plt.figure()         
set_spines()
plt.plot(AM_M3.AGE, AM_M3.COUNT, color = '#8EE9EE', marker = '*')
plt.plot(AM_F3.AGE, AM_F3.COUNT, color = '#FF4577', marker = '*')
auto_set_xlim(0,100) 
plt.savefig('./src/figAM3.png', format='png', bbox_inches='tight',
            transparent=True, dpi=300) 
plt.show()



# =============================================================================
# Masked HeatMap
# DEP => fig_DEP1_DEP2
# =============================================================================

# :
DEP = pd.DataFrame(data[['DEPNAIS1','DEPNAIS2']])
dep_stat = DEP.apply(pd.value_counts)
dep_corr = DEP.corr()

possible_DEP = DEP.DEPNAIS1.drop_duplicates().sort_values(ascending=True)
init_DEP = np.zeros((len(possible_DEP),len(possible_DEP)))

Matrix_DEP = pd.DataFrame(init_DEP)
Matrix_DEP.rename(columns=possible_DEP, 
                  index=possible_DEP, 
                  inplace=True)

####################################################################notfinished

# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================


