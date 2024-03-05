#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 17:25:29 2020

@author: marcusbe
"""


#Import Packgages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import matplotlib as mpl


#Pointwise Tax liability functon
def tax(x, tax_code = "linear", a = 60000):
    #Initially set tax liability to 0
    liability = 0
    #Round down to the next integer
    x = int(x)
    
    if tax_code == "none":
        liability = 0
 
    elif tax_code == "linear":
        tau = 0.3
        liability = tau * x
    
    elif tax_code == "abgeltung":
        tau = 0.25
        # tau = 0.28 #with church tax and solidarity surcharge
        F = 801 # for singles
        #F = 1602 #for couples
        liability = tau * max( x - F, 0)
    
    elif tax_code == "income":
        #German national income tax (§32a EStG)
        if x <= 9408:
            liability = 0
        elif x <= 14532:
            y = 1/10000 * ( x - 9408 ) 
            liability = (972.87 * y + 1400) * y
        elif x <= 57051:
            z = 1/10000 * ( x - 14532)
            liability = (212.02 * z + 2397) * z + 972.79
        elif x <= 270500:
            liability = 0.42 * x - 8963.74
        else:
            liability =  0.45 * x - 18078.74
    
    elif tax_code == "federal":
        #FX-Rate
        EURUSD = 1.12
        #Tax Base in USD
        x = x * EURUSD
        if x<= 9525:
            liability = 0.1 * max(x, 0) 
        elif x <= 38700:
            liability = 0.1 * 9525
            liability += 0.12 * ( x - 9525 ) 
        elif x<= 82500:
            liability = 0.1 * 9525
            liability += 0.12 * ( 38700 - 9525 ) 
            liability += 0.22 * ( x - 38700 )
        elif x <= 157500:
            liability = 0.1 * 9525
            liability += 0.12 * ( 38700 - 9525 ) 
            liability += 0.22 * ( 82500 - 38700 )
            liability += 0.24 * ( x - 82500 )
        elif x <= 200000:
            liability = 0.1 * 9525
            liability += 0.12 * ( 38700 - 9525 ) 
            liability += 0.22 * ( 82500 - 38700 )
            liability += 0.24 * ( 157500 - 82500 )
            liability += 0.32 * ( x - 157500 )
        elif x <= 500000:
            liability = 0.1 * 9525
            liability += 0.12 * ( 38700 - 9525 ) 
            liability += 0.22 * ( 82500 - 38700 )
            liability += 0.24 * ( 157500 - 82500 )
            liability += 0.32 * ( 200000- 157500 )
            liability += 0.35 * ( x - 200000 ) 
        else:
            liability = 0.1 * 9525
            liability += 0.12 * ( 38700 - 9525 ) 
            liability += 0.22 * ( 82500 - 38700 )
            liability += 0.24 * ( 157500 - 82500 )
            liability += 0.32 * ( 200000- 157500 )
            liability += 0.35 * ( 500000 - 200000 ) 
            liability += 0.37 * ( x - 500000 ) 
            
        #Transforming USD liability back to EUR
        liability = liability / EURUSD
    
    elif tax_code == "iso-residual":
        a = 3.40787
        rho = 0.865339
        liability = x - a * ( x ** rho )
        
    elif tax_code == "iso-residual-transformed":
        a = 3.40787
        rho = 0.865339
        if x <= a ** (1/(1 - rho)):
            liability = 0
        else: 
            liability = x - a * ( x ** rho )
    elif tax_code == "Robin-Hood":
        #a = 5000 * 12 #Average yearly net wealth  (Source: HFCS Report 2017)
        liability = x - a
    else:
        print("Tax liability function not defined. No tax" +
                                          "liability calculated!")
    
    return liability

# Marginal Tax rate
def dtax(x, tax_code = "linear"):
    return tax( x + 1, tax_code) - tax( x , tax_code)

# Test tax liabilty function
#print(tax(500000, tax_code = "abgeltung", tau = 0.3, F = 801))
# print(dtax(600000, tax_code = "federal", tau = 0, F = 0))

# Plot German income tax
bmg = np.linspace(0, 300000, 100)
tax_liability = np.array([tax(x, tax_code = "income") for x in bmg])
marginal_tax_rate = np.array([dtax(x, tax_code = "income") for x in bmg])

plt.plot(bmg, tax_liability)
plt.title("German Income Tax (§32a EStG)")
plt.xlabel("Tax Base in EUR")
plt.ylabel("Tax Liability in EUR")
plt.show()

plt.plot(bmg, marginal_tax_rate)
plt.title("German Income Tax (§32a EStG)")
plt.xlabel("Tax Base in EUR")
plt.ylabel("Marginal Tax Rate")
plt.show()

# Plot American Federal Income Tax
bmg = np.linspace(0, 600000, 100)
tax_liability = np.array([tax(x, tax_code = "federal") for x in bmg])
marginal_tax_rate = np.array([dtax(x, tax_code = "federal") for x in bmg])

plt.plot(bmg, tax_liability)
plt.title("American Federal Income Tax")
plt.xlabel("Tax Base in USD")
plt.ylabel("Tax Liability in USD")
plt.show()

plt.plot(bmg, marginal_tax_rate)
plt.title("American Federal Income Tax")
plt.xlabel("Tax Base in USD")
plt.ylabel("Marginal Tax Rate")
plt.show()

# Distribution of income per household

#Import Data
df = pd.read_csv("./income_structure_germany.csv", sep = ";", decimal = ",")

print(df.head(6))

print(df.info())

# Net Income Distribution
plt.plot(df["Cum Obs"], df["Cum Net"])
plt.title("Net Income Distribution Germany")
plt.xlabel("Portion of Households in Percent")
plt.ylabel("Portion of Net Income in Percent")
plt.show()

# Net Wealth Distribution (HFCS)
plt.plot(df["Cum Obs"], df["Cum Net Wealth"])
plt.title("Net Income Distribution Germany")
plt.xlabel("Portion of Households in Percent")
plt.ylabel("Portion of Net Income in Percent")
plt.show()

bmg = np.array(df["Avg Yearly Gross Income"])
tax_liability = np.array([tax(x, tax_code = "income") for x in bmg])
marginal_tax_rate = np.array([dtax(x, tax_code = "income") for x in bmg])

df["Avg Tax Liability"] = tax_liability
df["Avg Tax Rate Synthetic"] = tax_liability / bmg
df["Avg Marginal Tax Rate"] = marginal_tax_rate

# Compare Tax Rates
plt.figure( figsize =( 8, 8) )
plt.plot(df["Income Class"], df["Avg Tax Rate"], label = "Avg Tax Rate (real)")
plt.plot(df["Income Class"], df["Avg Tax Rate Synthetic"], label = "Avg Tax Rate (synth.)")
plt.plot(df["Income Class"], df["Avg Marginal Tax Rate"], label = "Marg Tax Rate (synth.)")
plt.title("Avg. Tax Rates (real/synth.) and marginal taxrates (synth.)")
plt.xlabel("Income Class")
plt.ylabel("Tax Rate")
plt.legend()
plt.savefig("marginal_tax_rates_income.png", dpi=400)
plt.show()



# Plot Reltive Saving Rate per Class
df["Saving Rate"] = 1- df["Consumption"] / df["Avg Net Income"]

plt.figure( figsize =( 8, 8) )
plt.bar(df["Income Class"], df["Saving Rate"])
#plot avg saving rate for comparison
plt.plot(df["Income Class"], df["Saving Rate"].mean()*np.ones(len(df["Income Class"])),'r--')
plt.title("Saving Rate per Income Class")
plt.xlabel("Income Class")
plt.ylabel("Saving Rate")
plt.savefig("saving_rate.png", dpi=400)
plt.show()


# Annual Gross Income growth rate
g = 0.02845  #CAGR of  Gross oncome from 2010 - 2019 (Statista)

# Clientele Effects
g_mult = np.array([0, 0.01, 0.02, 0.05, 0.08, 0.1])

# Annual Capital growth rate (Nooe r = k)
k = 0.0728 # CAGR of MSCI Performance from 2010-2019 (Statista)

# Clientele Effects
k_mult = np.array([-0.01, 0.02, 0.03, 0.07, 0.09, 0.11])

# Abgeltungsteuer
tau = 0.25 #without Soli and Kist
#tau = 0.28 #with Soli and KiSt


k_net = k * (1 - tau)

k_mult_net = k_mult * (1 - tau)

gross_income = bmg
net_income = gross_income - tax_liability 
net_wealth = df["Avg Net Wealth (HFCS)"]
consumption = df["Consumption"] * 12

net_wealth = net_wealth + (net_income - consumption) 

obs = df["Obs"]
cum_obs = df["Cum Obs"]

total_net_wealth = np.sum(net_wealth * obs) #= net_wealth.T @ obs

rel_net_wealth = net_wealth * obs / total_net_wealth

cum_net_wealth = np.array([ np.sum(rel_net_wealth[:i]) 
                            for i in range(1, len(rel_net_wealth) + 1)])

# Net Wealth Distribution (HFCS)
plt.plot(cum_obs  , cum_net_wealth)
plt.title("Net Income Distribution Germany")
plt.xlabel("Portion of Households in Percent")
plt.ylabel("Portion of Net Income in Percent")
plt.savefig("lorenz-germany.png", dpi=400)

    
# =============================================================================
#   Final Model 
# =============================================================================

g = 0.02845 
k = 0.0728 

#g = 0.1
#k = 0

#g = g_mult
#k = k_mult
        
model = ["linear", "abgeltung", "income", "federal", "iso-residual", 
         "iso-residual-transformed", "Robin-Hood"]

#model = ["income"]

for model in model:
    
    compound_tax = np.zeros(len(df["Avg Yearly Gross Income"]))
    
    print("Model: ", model)
    plt.figure( figsize =( 10, 10) )
    
    for t in range(0, 10, 1):
                   
        if t == 0:
            bmg = np.array(df["Avg Yearly Gross Income"]) 
        else:
            bmg = bmg * (1 + g ) 
    
        tax_liability = np.array([ tax(x, tax_code = model, a = bmg.mean()) for x in bmg])
        
        #print(tax_liability)
        
        net_income = bmg - tax_liability 
        
        #print(net_income)
        
        if t== 0:
            net_wealth = df["Avg Net Wealth (HFCS)"] 
            consumption = df["Consumption"] * 12 
            #other = df["Other Expenses"] * 12 
            other = np.zeros(len(net_income))
            tax_cap = np.zeros(len(net_income))
        else:
            #Tax Base for Capital Tax
            delta_net_wealth = net_wealth * k 
            bmg_cap = delta_net_wealth  
            #print(bmg_cap)
            tax_cap =  np.array([ tax(x, tax_code = "abgeltung") for x in bmg_cap ]).real
            #print(tax_cap)
            net_wealth = net_wealth * (1 + k) 
            consumption = consumption * (1 + g) 
            other = other * (1 + g) 
        
        if model in ["Robin-Hood", "iso-residual"] :
            bmg_cap = net_wealth  - consumption - other 
            tax_cap =  np.array([ tax(x, tax_code = model, a = bmg_cap.mean()) for x in bmg_cap ]).real
        
        net_wealth = net_wealth + net_income - consumption - other - tax_cap
        
        #print(net_wealth)
        
        obs = df["Obs"]
        cum_obs = df["Cum Obs"]
        
        total_net_wealth = np.sum(net_wealth * obs) #= net_wealth.T @ obs
        #print("Total net wealth in %i (in Mrd.): %d"  % (t, total_net_wealth/1000000000))
        
        # Total Tax Revenues (income tax + capital tax)
        total_tax_revenues = np.sum(tax_liability  * obs) + np.sum(tax_cap * obs)
        #print("Tax revenues in %i (in Mrd.): %d"  % (t, total_tax_revenues/1000000000))
        
        compound_tax += (( tax_liability + tax_cap ) * obs)
        #print(compound_tax)
        compound_tax *= (1+g)
        #print(compound_tax)
        
        
        rel_net_wealth = (net_wealth * obs) / total_net_wealth
        
        #print( "Relative Net Wealth in %i: %s" % (t, str(rel_net_wealth)) )
        print( rel_net_wealth.values )
        
        cum_net_wealth = np.array([ np.sum(rel_net_wealth[:i]) 
                                    for i in range(1, len(rel_net_wealth) + 1)])
        
        # Net Wealth Distribution (HFCS)
        #plt.plot(net_wealth , label = " t = %i " % t)
        plt.plot(cum_obs  , cum_net_wealth, label = " t = %i " % t) 
        plt.title("Net Wealth Distribution: Model = %s" % model )
        plt.xlabel("Portion of Households in Percent")
        plt.ylabel("Portion of Net Income in Percent")
        plt.legend()
    print("Total Net Wealth in t = %i (in Bn.): %.2f" % (t, total_net_wealth/1000000000000))
    print("Compound tax revenues in t = %i (in Bn.): %.2f" % (t, np.sum(compound_tax)/1000000000000))
    print("Total Welfare in t = %i (in Bn.): %.2f" % (t, (total_net_wealth + np.sum(compound_tax))/1000000000000))
    plt.savefig("wealth_dstribution_"+ model +"_mult.png", dpi=400)
    plt.show()
#    mpl.use("pgf")
#    mpl.rcParams.update({
#                            "pgf.texsystem": "pdflatex",
#                            'font.family': 'serif',
#                            'text.usetex': True,
#                            'pgf.rcfonts': False,
#                        })


# =============================================================================
# Sensitivity Analysis
# =============================================================================
    
def inequality_index(T = 10, g = g, k = k, model = "income" , printing = False, measure = "rich quantile" ):
    
    # Inequality Coefficient 
    ineq = 0
    
    for t in range(0, T, 1):
                   
        if t == 0:
            bmg = np.array(df["Avg Yearly Gross Income"])
        else:
            bmg = bmg * ( 1 + g ) 
    
        tax_liability = np.array([ tax(x, tax_code = model, a = bmg.mean()) for x in bmg])
               
        net_income = bmg - tax_liability 
               
        if t== 0:
            net_wealth = df["Avg Net Wealth (HFCS)"] 
            consumption = df["Consumption"] * 12  
            #consumption = np.zeros(len(net_income))
            other = np.zeros(len(net_income))
            tax_cap = np.zeros(len(net_income))
        else:
            #Tax Base for Capital Tax
            delta_net_wealth = net_wealth * k 
            bmg_cap = delta_net_wealth  
            tax_cap =  np.array([ tax(x, tax_code = "abgeltung") for x in bmg_cap ]).real
            net_wealth = net_wealth * (1 + k) 
            consumption = consumption * (1 + g) 
            other = other * (1 + g) 
        
        if model in ["Robin-Hood", "iso-residual"] :
            bmg_cap = net_wealth  - consumption - other 
            tax_cap =  np.array([ tax(x, tax_code = model, a = bmg_cap.mean()) for x in bmg_cap ]).real
        
        net_wealth = net_wealth + net_income - consumption - other - tax_cap
        
        obs = df["Obs"]
        
        total_net_wealth = np.sum(net_wealth * obs) #= net_wealth.T @ obs
        
        
        rel_net_wealth = (net_wealth * obs) / total_net_wealth
        
        rel_net_wealth = rel_net_wealth.values
        
        if printing: 
            print( rel_net_wealth )
    # Rel. Wealth last Quantile / Rel. Net Wealth First Quantile  
    # The higher the inequality coefficient the more unequally 
    # distributed the net wealth 
    #i
    if measure == "rich quantile":
        ineq = rel_net_wealth[-1]
    elif measure == "poor quantile":
        ineq = rel_net_wealth[0]
    elif measure == "inequality coefficient":
        ineq = rel_net_wealth[-1] / rel_net_wealth[0]
    else:
        print("Inequality measure does not exist")
        
    return ineq
    

# Test 
test = inequality_index(T = 1, g = g, k = k, model = "income", printing = "True", measure = "rich quantile")
print(test)

# 3D Surfacte Plot of different g and k levels

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

        
model = ["linear", "abgeltung", "income", "federal", "iso-residual", 
         "iso-residual-transformed", "Robin-Hood"]

#model = "federal"

measure = "rich quantile"

# Number of datapoints
n = 10

#Limits
maxG = 0.3
maxK = 0.3

#End of Period
T = 10

for model in model:

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    # Make data.
    G = np.arange(0, maxG, maxG / n)
    K = np.arange(0, maxK, maxK / n)
            
    # Indequallity Index
    I = np.array([ [inequality_index(T = T, g = g, k = k, model = model, measure = measure ) for g in G ] for k in K])
    
    if model == "Robin-Hood":
        # Smootjing the calculation errors
        I = np.round(I, 2)
    
    G, K = np.meshgrid(G, K)
    
    # Plot the surface.
    surf = ax.plot_surface(G, K, I, cmap = cm.coolwarm,
                           linewidth = 0, antialiased = False)
    
    
    plt.title("Inequality Coefficient: Model = %s, T = %i" % (model, T - 1) )
    # Customize the z axis.
    #ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.view_init(20, 200)
    
    ax.set_xlabel("g")
    ax.set_ylabel("r")
    ax.set_zlabel(measure)
    
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    # Save figure
    plt.savefig("inequality_index_"+ model +".png", dpi = 400)
    plt.show()


#Plot Single g

fig = plt.figure()

#g = 0.02845 
k = 0.0728 

# Make data.
G = np.arange(0, 0.1, 0.01)

model = "income"
measure = "rich quantile"
        
# Indequallity Index
I = np.array( [inequality_index(T = T, g = g, k = k, model = model, measure = measure ) for g in G ] )

plt.plot(G,I)

plt.show()


# Plot Single k

fig = plt.figure()

g = 0.02845 
#k = 0.0728 

# Make data.
K = np.arange(0, 0.1, 0.01)

model = "income"
measure = "rich quantile"
        
# Indequallity Index
I = np.array( [inequality_index(T = T, g = g, k = k, model = model, measure = measure ) for k in K ] )

plt.plot(K,I)

plt.show()


    