import math
import torch
import turtle


class Flight():
    
    def __init__(self, orig, dest, delta_t, v):  #orig/dest des tenseurs (1,2)
        
        self.orig = orig
        self.dest = dest
        self.delta_t = delta_t
        self.v=v
        
        self.u = self.dest - self.orig  #vecteur
        self.distance = math.sqrt(float(torch.sum(self.u**2))) #distance la plus courte a parcourir 
        self.delta_d = self.v*delta_t
        self.N = self.distance/self.delta_d
        self.cap = math.atan2(self.u[0],self.u[1]) #cap (par rapport au nord)
        
        self.u_reg = self.u/(self.N-1) #u de longueur delta_d
        self.u_clone = torch.clone(self.u_reg)
        
        self.t_range = torch.arange(self.N).float() #torch.tensor([0,1,2,3,...,longueur-1])
        self.liste_temps = self.t_range*self.delta_d/v #cree un vecteur des temps
        self.pas_de_temps = self.liste_temps[1] - self.liste_temps[0] #calcul du pas de temps        
        
        self.trajectoire = self.u_reg*self.t_range.resize(int(self.N),1) + self.orig #calul de la trajectoire
        self.trajectoire_init = torch.clone(self.trajectoire) #a faire pour ne pas a recalculer a chaque fois 
        
        #dessin turtle
        self.avion = turtle.Turtle("turtle")
        self.avion.color("green","blue")
        self.avion.up()
        self.avion.setpos(400*float(orig[0])/self.distance, 400*float(orig[1])/self.distance)
        self.avion.down()
        self.avion.speed("slow")
        self.avion.right(self.cap*180/math.pi - 90)
        
        
    def plot_turtle(self,temps):
        
        size = self.trajectoire.size()[0]
        x = [float(self.trajectoire[i,0]) for i in range(size)]
        y = [float(self.trajectoire[i,1]) for i in range(size)]
        self.avion.setpos(400*x[int(temps*self.v/self.delta_d)]/self.distance, 400*y[int(temps*self.v/self.delta_d)]/self.distance)
        
        
    def plot_graph(self):
        
        size = self.trajectoire.size()[0]
        x = [self.trajectoire[i,0] for i in range(size)]
        y = [self.trajectoire[i,1] for i in range(size)]
        return x,y

