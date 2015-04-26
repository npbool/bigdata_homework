import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt

def load_data(col):
	f = open("data.csv", 'r')
	lines = f.read().split('\r')
	
	cat = [int(line.split(',')[1]) for line in lines[1:]]
	age = [float(line.split(',')[col]) for line in lines[1:]]

	age_comp = [[] for i in range(5)]
	for c,a in zip(cat,age):
		age_comp[c-1].append(a)

	f.close()
	return np.array(age), [np.array(c) for c in age_comp]

def normality_test(data):
	k2,pv = stats.mstats.normaltest(data)	

	mean = sum(data)/len(data)
	var = (data-mean).dot(data-mean)/(len(data)-1)
	std = np.sqrt(var)

	kde = sp.stats.gaussian_kde(data)
	lower = data.min()/2

	x = np.linspace(lower, data.max(), 1000)
	p = kde(x)
	norm_var = stats.norm
	
	p_norm = 1/np.sqrt(2*np.pi*var) * np.exp( -(x-mean)*(x-mean)  /2 /var)

	#p: estimated pdf using kernel density estimation
	#p_norm: pdf of Gaussian with same mean and variance
	#pv: P value of normality test
	return x,p,p_norm,pv

def recal_mean_std(x,p):
	mean = 0
	var = 0
	for i in range(len(x)-1):
		var += (x[i+1]-x[i])*p[i]*x[i]*x[i]
		mean += (x[i+1]-x[i])*p[i]*x[i]
	var -= mean*mean
	return mean,np.sqrt(var)

def visualize(data, data_comp, name):
	colors = 'bgrmy'
	f = plt.figure()	

	plt.subplot(211)
	plt.xlabel(name)	
	for i,comp in enumerate(data_comp):		
		x,p,p_norm,pv = normality_test(comp)
		mean = np.mean(x)
		stddev = np.std(x)		
		plt.plot(x, p, colors[i], label="group %d" %(i+1) )
		#print mean, stddev
		#mean, stddev = recal_mean_std(x,p)
	plt.legend()
	plt.subplot(212)		
		#plt.errorbar(i, mean, yerr=stddev, ecolor=colors[i])
	grand_mean = np.mean(data)
	plt.boxplot(data_comp)
	plt.axhline(y=grand_mean, label="Grand Mean")
	plt.xlabel("Group")
	plt.ylabel(name)
	plt.legend()
	plt.savefig("img/"+"vis_"+name+".png")	

# Draw pdf for whole data and each group
# Normality Test
# Check variance ratio
def test_anova(data, data_cat, name):				
	print "Test for %s" % name
	#normality test with all data
	x,p,p_norm,pv = normality_test(data)
	plt.figure()
	plt.plot(x, p, 'r', label="estimate pdf")
	plt.plot(x, p_norm, 'b--', label="gaussian pdf")
	plt.title("P value:" + "%.4e"%pv)
	plt.legend()
	plt.savefig("img/"+"grand_%s.png" % name)
	print "Normality test P value: ", pv
	#normality test within each group
	f,ax = plt.subplots(5,1,sharex=True,figsize=(8,12))	

	group_pv = []
	for i,comp in enumerate(data_cat):				
		ax[i].set_title("Group " + str(i))

		x,p,p_norm,pv = normality_test(comp)
		ax[i].plot(x, p, 'r', label="estimate pdf")
		ax[i].plot(x, p_norm, 'b--', label="gaussian pdf")
		ax[i].legend(loc=1, prop={'size':10})	
		group_pv.append(pv)
	f.savefig("img/"+"groups_%s.png" % name)
	
	variances = [np.var(comp) for comp in data_cat]
	stds = [np.std(comp) for comp in data_cat]
	max_std = max(stds)
	min_std = min(stds)
	
	print "Group Variances"
	for s in variances:
		print "%.2f"%s, '\t',
	print ''
	print "Group pv"
	for s in group_pv:
		print str(s)+ '\t',
	print ''
	print "Group Std. Dev."
	for s in stds:
		print "%.2f"%s, '\t',
	print ''
	print "Max std.dev %.2f"%max_std
	print "Min std.dev %.2f"% min_std
	print "Max/Min std deviation ratio"
	ratio = max_std/min_std
	print ratio

	#one-way anova
def ANOVA(data_cat):
	fv, pv = stats.f_oneway(*data_cat)
	print "one-way anova:"
	print "F value:" + str(fv)
	print "P value:" + str(pv)

def log(data_comp):
	return [np.log10(d) for d in data_comp]
def compare_log(data, data_cat,name):
	f,ax = plt.subplots(2,1)
	plt.title(name)
	x,p,p_norm,pv = normality_test(data)
	ax[0].plot(x, p, 'r', label="estimate pdf")
	ax[0].plot(x, p_norm, 'b--', label="gaussian pdf")
	ax[0].set_title(name+" pdf")
	ax[0].legend()
	#ax.savefig(""img/"+grand.png")

	x,p,p_norm,pv = normality_test(np.log10(data))
	ax[1].plot(x, p, 'r', label="estimate pdf")
	ax[1].plot(x, p_norm, 'b--', label="gaussian pdf")
	ax[1].set_title("Log transformation pdf")

	#p: estimated pdf using kernel density estimation
#p_norm: pdf of Gaussian with same mean and variance
#pv: P value of normality test
	plt.savefig("img/" + name+".png")

ages, age_cat = load_data(6)
# Draw pdf for whole data and each group
test_anova(ages, age_cat, "Age") #age
ANOVA(age_cat)
visualize(ages, age_cat, "Age")

print ""
msg, msg_cat = load_data(3)
compare_log(msg, msg_cat, "Message")
msg_log = np.log10(msg)
msg_cat_log = [np.log10(comp) for comp in msg_cat]
test_anova(msg_log, msg_cat_log, "logMessage")

print ""
mem, mem_cat = load_data(2)
test_anova(mem, mem_cat, "Member")
compare_log(mem, mem_cat, "Member")
test_anova(np.log10(mem), log(mem_cat), "log Member" )

print ""
conv, conv_cat = load_data(10)
test_anova(conv, conv_cat, "Conversation")
compare_log(conv, conv_cat, "Conversation")
test_anova(np.log10(conv), log(conv_cat), "log Conversation" )

print "Non parametric ANOVA -- Kurskal-Wallis H test"
H, pv = sp.stats.kruskal(*msg_cat)
print "Message"
print "Kruskal H ", H, ", p ", pv
H, pv = sp.stats.kruskal(*log(msg_cat))
print "Kruskal H for log ", H, ", p ", pv
visualize(np.log10(msg), log(msg_cat), "log Message")

print ""
H, pv = sp.stats.kruskal(*mem_cat)
print "Member"
print "Kruskal H ", H, ", p ", pv
visualize(np.log10(mem), log(mem_cat), "log Memeber")

print ""
H, pv = sp.stats.kruskal(*conv_cat)
print "Conversation"
print "Kruskal H ", H, ", p ", pv
visualize(np.log10(conv), log(conv_cat), "log Conversation")