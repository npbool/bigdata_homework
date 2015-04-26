import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt

def load_data(col=6):
	f = open("data.csv", 'r')
	lines = f.read().split('\r')
	
	cat = [int(line.split(',')[1]) for line in lines[1:]]
	age = [float(line.split(',')[col]) for line in lines[1:]]

	age_comp = [[] for i in range(5)]
	for c,a in zip(cat,age):
		age_comp[c-1].append(a)
	#data = [float(line.decode('utf8').split(u',')[6]) for line in f]
	#print data
	f.close()
	return np.array(age), [np.array(c) for c in age_comp]

def plot_epdf(data):
	k2,pv = stats.mstats.normaltest(data)	

	mean = sum(data)/len(data)
	var = (data-mean).dot(data-mean)/(len(data)-1)
	std = np.sqrt(var)
	#print "Mean:", mean
	#print "Var:", var
	#print "PV:", pv

	kde = sp.stats.gaussian_kde(data)
	x = np.linspace(data.min(), data.max(), 100)
	p = kde(x)
	norm_var = stats.norm
	#p_norm = norm_var.pdf((x-mean)/np.sqrt(var))
	p_norm = 1/np.sqrt(2*np.pi*var) * np.exp( -(x-mean)*(x-mean)  /2 /var)
	return x,p,p_norm,pv

def analyze_age():			
	data, data_comp = load_data()

	#normality test with all data
	x,p,p_norm,pv = plot_epdf(data)
	plt.figure()
	plt.plot(x, p, 'r', label="estimate pdf")
	plt.plot(x, p_norm, 'b--', label="gaussian pdf")
	plt.title("P value:" + "%.4e"%pv)
	plt.legend()
	plt.savefig("grand.png")
	
	#normality test within each group
	f,ax = plt.subplots(5,1,sharex=True,figsize=(8,12))	
	for i,comp in enumerate(data_comp):				
		x,p,p_norm,pv = plot_epdf(comp)		
		ax[i].set_title("Group " + str(i)+", P value:" + "%.4e"%pv)
		ax[i].plot(x, p, 'r', label="estimate pdf")
		ax[i].plot(x, p_norm, 'b--', label="gaussian pdf")
		ax[i].legend(loc=1, prop={'size':10})							
	f.savefig("groups.png")

	
	variances = [np.var(comp) for comp in data_comp]
	stds = [np.std(comp) for comp in data_comp]
	max_std = max(stds)
	min_std = min(stds)
	ratio = max_std/min_std
	print "Average Age Group Variances"
	for s in variances:
		print "%.2f"%s, ' ',
	print ''
	print "Average Age Group Std. Dev."
	for s in stds:
		print "%.2f"%s, ' ',
	print ''
	print "Max/Min std deviation ratio"
	print ratio

	#one-way anova
	fv, pv = stats.f_oneway(*data_comp)
	print "one-way anova:"
	print "F value:" + str(fv)
	print "P value:" + str(pv)
def analyze_column(col,name,filename):
	data, data_comp = load_data(col)		

	f,ax = plt.subplots(2,1)
	plt.title(name)
	x,p,p_norm,pv = plot_epdf(data)
	ax[0].plot(x, p, 'r', label="estimate pdf")
	ax[0].plot(x, p_norm, 'b--', label="gaussian pdf")
	ax[0].set_title(name+" pdf")
	ax[0].legend()
	#ax.savefig("grand.png")

	x,p,p_norm,pv = plot_epdf(np.log(data))
	ax[1].plot(x, p, 'r', label="estimate pdf")
	ax[1].plot(x, p_norm, 'b--', label="gaussian pdf")
	ax[1].set_title("Log transformation pdf")
	ax[1].legend()

	plt.savefig(filename)
analyze_age()
#analyze_column(3, "number of message", "nmessage.png")
#analyze_column(2, "number of members", "nmember.png")
#analyze_column(10, "number of conversations", "nconv.png")