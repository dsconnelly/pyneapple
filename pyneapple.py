# ==========================================================
# this file contains PYNEAPPLE:
# PYthon for Near-Earth AtmosPheric Predictive LEarning
# ==========================================================

# ==========================================================
# import statements: numpy, matplotlib with Basemap,
# netCDF4, sklearn, and some math functions
# ==========================================================
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as prep
from math import cos,radians

# ==========================================================
# constants for use in determining gridbox areas
# ==========================================================
_latm = 110950
_lonm = 111320

# ==========================================================
# here we grab latitude and longitude arrays for plotting
# ==========================================================
with Dataset('data/historical_averages.nc',mode='r') as g:
	lat = g.variables['lat'][:]
	lon = g.variables['lon'][:]
lon = np.insert(lon,144,360)
x,y = np.meshgrid(lon,lat)

# ==========================================================
# global_avg() returns an average of a 73x144 array
# weighted by the area of each gridbox (used to compute
# global mean temperatures)
# ==========================================================
def global_avg(a):
	output = 0
	for i in range(0,73):
		for j in range(0,144):
			output += a[i][j] * _latm * _lonm * 2.5 * 2.5 * (cos(radians(lat[i])))
	output = output / (5.1 * (10 ** 14))
	return output
			
# ==========================================================
# run_pyneapple() runs the given model on the given array
# and returns the output already placed in a 73x144 gridbox
# ==========================================================
def run_pyneapple(input,model):
	pre = model.predict(input)
	air_t = np.zeros((73,144))
	c = 0
	for i in range(0,73):
		for j in range(0,144):
			air_t[i][j] = pre[c]
			c = c + 1
			
	return air_t
	
# ==========================================================
# plot_up() plots the data with the parameters given
# ==========================================================
def plot_up(data,heading,units,scale,fname=None,col='jet'):
	plt.figure(heading)
	
	m = Basemap(projection='robin', resolution='l', lon_0=0)
	m.drawcoastlines()
	m.drawparallels(np.arange(-90.,120.,30.))
	m.drawmeridians(np.arange(0.,360.,60.))

	chart = m.pcolormesh(x,y,data,latlon=True,cmap=col,norm=scale)
	legend = m.colorbar(chart,location='bottom',pad='10%')
	legend.set_label(units)
	plt.title(heading)
	
	if fname is None:
		plt.show()
	else:
		plt.savefig(fname)
		
# ==========================================================
# these dictionaries hold historical data on the features
# ==========================================================
carbon = {1950 : 311.3, 1970 : 325.54, 1980 : 338.8, 1990 : 353.96, 2010 : 388.56}
methane = {1950 : 1147, 1970 : 1386, 1980 : 1547.75, 1990 : 1713.99, 2010 : 1798.62}
nitrous = {1950 : 289, 1970 : 295.2, 1980 : 301.382, 1990 : 309.485, 2010 : 323.071}
population = {1950 : 2.52, 1970 : 3.7, 1980 : 4.45, 1990 : 5.31, 2010 : 6.9}

# ==========================================================
# these dictionaries contain feature data for RCP scenarios
# ==========================================================
RCP6 = {'carbon' : 669.723, 'methane' : 1649.396, 'nitrous' : 406.625, 'population' : 9.5}
RCP8 = {'carbon' : 935.874, 'methane' : 3750.685, 'nitrous' : 435.106, 'population' : 12.0}

# ==========================================================
# we assemble the training arrays
# ==========================================================
train_in = np.zeros((31536,5))
train_out = np.zeros(31536)
s = 0
with Dataset('data/historical_averages.nc',mode='r') as g:
	for i in range(0,73):
		for j in range(0,144):
			t = g.variables['air_t_long'][:][0][i][j]
		
			train_in[s][0] = carbon[1970] / carbon[1950] * t
			train_in[s + 10512][0] = carbon[1980] / carbon[1950] * t
			train_in[s + 10512][0] = carbon[1990] / carbon[1950] * t
			
			train_in[s][1] = methane[1970] / methane[1950]
			train_in[s + 10512][1] = methane[1980] / methane[1950]
			train_in[s + 10512][1] = methane[1990] / methane[1950]
			
			train_in[s][2] = nitrous[1970] / nitrous[1950]
			train_in[s + 10512][2] = nitrous[1980] / nitrous[1950]
			train_in[s + 10512][2] = nitrous[1990] / nitrous[1950]
			
			train_in[s][3] = population[1970] / population[1950]
			train_in[s + 10512][3] = population[1980] / population[1950]
			train_in[s + 10512][3] = population[1990] / population[1950]
			
			train_in[s][4] = t
			train_in[s + 10512][4] = t
			train_in[s + 21024][4] = t
			
			train_out[s] = g.variables['air_t_long'][:][1][i][j]
			train_out[s + 10512] = g.variables['air_t_long'][:][2][i][j]
			train_out[s + 21024] = g.variables['air_t_long'][:][4][i][j]
			
			s = s + 1
			
# ==========================================================
# we construct tests for 2010, RCP6, and RCP8.5 (year 2100)
# (the RCP8.5 array is called test_rcp8 to avoid periods)
# ==========================================================			
test_2010 = np.zeros((10512,5))
test_rcp6 = np.zeros((10512,5))
test_rcp8 = np.zeros((10512,5))
s = 0
with Dataset('data/historical_averages.nc',mode='r') as g:
	for i in range(0,73):
		for j in range(0,144):
			t = g.variables['air_t_long'][:][0][i][j]
		
			test_2010[s][0] = carbon[2010] / carbon[1950] * t
			test_2010[s][1] = methane[2010] / methane[1950]
			test_2010[s][2] = nitrous[2010] / nitrous[1950]
			test_2010[s][3] = population[2010] / population[1950]
			test_2010[s][4] = t
			
			test_rcp6[s][0] = RCP6['carbon'] / carbon[1950] * t
			test_rcp6[s][1] = RCP6['methane'] / methane[1950]
			test_rcp6[s][2] = RCP6['nitrous'] / nitrous[1950]
			test_rcp6[s][3] = RCP6['population'] / population[1950]
			test_rcp6[s][4] = t
			
			test_rcp8[s][0] = RCP8['carbon'] / carbon[1950] * t
			test_rcp8[s][1] = RCP8['methane'] / methane[1950]
			test_rcp8[s][2] = RCP8['nitrous'] / nitrous[1950]
			test_rcp8[s][3] = RCP8['population'] / population[1950]
			test_rcp8[s][4] = t
					
			s = s + 1
		
# ==========================================================
# we normalize each sample
# ==========================================================
train_in = prep.normalize(train_in)
test_2010 = prep.normalize(test_2010)
test_rcp6 = prep.normalize(test_rcp6)
test_rcp8 = prep.normalize(test_rcp8)

# ==========================================================
# we create the pyneapple model and train it
# ==========================================================		
pyneapple = RandomForestRegressor(n_estimators=40)
pyneapple.fit(train_in,train_out)

# ==========================================================
# we run pyneapple on 2010, RCP6, and RCP8.5 data
# ==========================================================
out_2010 = run_pyneapple(test_2010,pyneapple)
out_rcp6 = run_pyneapple(test_rcp6,pyneapple)
out_rcp8 = run_pyneapple(test_rcp8,pyneapple)

# ==========================================================
# using global_avg() we calculate the average temperatures
# that pyneapple predicts for each dataset
# ==========================================================
print('mean 2010 temperature is ' + str(global_avg(out_2010))[:5] + ' C')
print('mean RCP6 temperature is ' + str(global_avg(out_rcp6))[:5] + ' C')
print('mean RCP8.5 temperature is ' + str(global_avg(out_rcp8))[:5] + ' C')

# ==========================================================
# we pull the real 2010 data for comparison
# ==========================================================
with Dataset('data/historical_averages.nc',mode='r') as g:
	real_air_t_2010 = g.variables['air_t_long'][:][5]

# ==========================================================
# here are three useful norms:
#      1. tnorm for plotting global temperatures
#      2. bnorm for plotting ratios
#      3. dnorm for plotting differences between years
# ==========================================================
tnorm = colors.Normalize(vmin=-35,vmax=35,clip=True)
bnorm = colors.BoundaryNorm([0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.95,1.0,1.05,1.1,1.15,1.2,1.4,1.6,1.8,2.0],256,clip=True)
dnorm = colors.Normalize(vmin=0,vmax=10,clip=True)

# ==========================================================
# task 1: plot 2010 prediction, real 2010 data, and ratios
# ==========================================================
plot_up(out_2010,'2010 Predicted Air Temperature','C',tnorm,'plots/2010/prediction.png')
plot_up(real_air_t_2010,'2010 Real Air Temperature','C',tnorm,'plots/2010/real.png')
plot_up(out_2010/real_air_t_2010,'2010 Air Temperature Model Accuracy','predicted/real',bnorm,'plots/2010/comparison.png','RdBu_r')

# ==========================================================
# task 2: plot RCP6 prediction and plot warming
# between 2010 and 2100 under RCP6
# ==========================================================
plot_up(out_rcp6,'2100 Predicted Air Temperature (RCP6)','C',tnorm,'plots/rcp6/prediction.png')
plot_up(out_rcp6 - out_2010,'Difference Between 2100 (RCP6) and 2010','C',dnorm,'plots/rcp6/difference.png')

# ==========================================================
# task 3: plot RCP8.5 prediction and plot warming
# between 2010 and 2100 under RCP8.5
# ==========================================================
plot_up(out_rcp8,'2100 Predicted Air Temperature (RCP8.5)','C',tnorm,'plots/rcp8/prediction.png')
plot_up(out_rcp8 - out_2010,'Difference Between 2100 (RCP8.5) and 2010','C',dnorm,'plots/rcp8/difference.png')