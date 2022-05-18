import numpy as np
import torch


# Liste de fonctions à utiliser pour tester l'efficacité/précision de l'algorithme d'évolution différentielle
#N.B.: 1) VTR = Value to Reach. On considère que l'optimisation est faite si le minimum trouvé < VTR
#      2) Les fonctions suivantes prennent:
#         - soit une population transposé (tensor n x NP) et renvoient une ligne (tensor 1 x NP) 
#         - soit un vecteur tensor(1 x n)
           
            
def ackley(vct):
    """
    Ackley function
    Domain: [-5,5] 
    Dimension: 2
    Real minimum: f(0,0) = 0
    """
    x = vct[0]
    y = vct[1]
    return -20*torch.exp(-0.2*(0.5*(x**2+y**2))**0.5) - torch.exp(0.5*(torch.cos(2*np.pi*x) + torch.cos(2*np.pi*y))) + np.exp(1) + 20

def beale(vct):
    """
    Beale function
    Domain: [-4.5,4.5] 
    Dimension: 2
    Real minimum: f(3,0.5) = 0
    """
    x = vct[0]
    y = vct[1]
    return (1.5-x+x*y)**2 + (2.25-x+x*y**2)**2 + (2.625-x+x*y**3)**2 

def goldstein(vct):
    """
    Goldstein–Price function
    Domain: [-2,2]
    Dimension: 2
    Real minimum: f(0,-1) = 3
    """
    x = vct[0]
    y = vct[1]
    return (1+(x+y+1)**2*(19-14*x+3*x**2-14*y+6*x*y+3*y**2))*(30+(2*x-3*y)**2*(18-32*x+12*x**2+48*y-36*x*y+27*y**2))
    
def booth(vct):
    """
    Booth function
    Domain: [-10,10]
    Dimension: 2
    Real minimum: f(1,3) = 0
    """
    x = vct[0]
    y = vct[1]
    return (x+2*y-7)**2 + (2*x+y-5)**2

def levi(vct):
    """
    Lévi function n.13
    Domain: [-10,10]
    Dimension: 2
    Real minimum: f(1,1) = 0
    """
    x = vct[0]
    y = vct[1]
    return torch.sin(3*np.pi*x)**2+(x-1)**2*(1+torch.sin(3*np.pi*y)**2)+(y-1)**2*(1+torch.sin(2*np.pi*y)**2)

def camel(vct):
    """
    Three-hump camel function
    Domain: [-5,5]
    Dimension: 2
    Real minimum: f(0,0) = 0
    """
    x = vct[0]
    y = vct[1]
    return 2*x**2 - 1.05*x**4 + x**6/6 + x*y + y**2

def easom(vct):
    """
    Easom function
    Domain: [-10,10]
    Dimension: 2
    Real minimum: f(pi,pi) = -1
    """
    x = vct[0]
    y = vct[1]
    return -torch.cos(x)*torch.cos(y)*torch.exp(-((x-np.pi)**2+(y-np.pi)**2))

def dejong1(vct):
    """
    Domain: [-5,12,5.12]
    Dimension: 3
    VTR = 1.e-6
    Real minimum: f(0,0,0) = 0
    """
    return sum(vct[i]**2 for i in range(3))

def dejong2(vct):
    """
    Domain: [-2,2]
    Dimension: 2
    VTR= 1.e-6
    Real minimum: f(1,1) = 0
    """
    x = vct[0]
    y = vct[1]
    return 100*(x**2-y)**2 + (1-x)**2 

def dejong3(x):
    """
    Domain: [-5.12,5.12]
    VTR= 1.e-6
    Real minimum: f(-5-eps) = 0 où 0 < epsilon < 0.012
    """
    n = x.size()[0]  
    x = x.float()
    expression1 = x.clone().apply_(lambda e : np.floor(e))
    expression2 = x.clone().apply_(lambda e : 30 * (e - 5.12))
    expression3 = x.clone().apply_(lambda e : 30 * (5.12 - e))

    condition1 = abs(x) <= 5.12 #(popu > -5.12) & (popu < 5.12)
    transformee = torch.where(condition1, expression1, expression3 )    
    condition2 = x > 5.12
    transformeefinal = torch.where(condition2, expression2, transformee)

    return sum(transformeefinal[i] for i in range(n))

def dejong4(vct):
    """
    Domain: [-1.28,1.28]
    Dimension: 30
    VTR = 15
    Real minimum: Unknown. Cf VTR.
    """
    return sum(vct[i]**4*i + np.random.normal(0,0.1,1)[0] for i in range(30))

def dejong5(vct): 
    """
    Domaine=[-65.536,65.536]
    VTR = 0.998005
    Dimension: 2
    Real minimum: f(-32,-32) = 0.998004
    """
    def function(i,j):
        l = [-32, -16, 0, 16, 32]
        if j==1:
            i = i%5
            return l[i]
        else:
            i = i//5
            return l[i]
            
    return 1/(0.002 + sum(1/(i + sum((vct[j-1]-function(i,j))**6 for j in range (1,3))) for i in range(25))) 

