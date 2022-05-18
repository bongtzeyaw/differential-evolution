import math
import torch
import algo
import flights
import conflicts
import matplotlib.pyplot as plt
import turtle
import time


CR = 0.75
F = 1.5
NP = 50
NG = 50

DELTA_T = 0.02 #pas de temps
VITESSE = 10 #vitesse
SEPARATION = 20 #distance securité avion
N_AVIONS = 5
RAYON = 100 #rayon_cercle
 

def f_isocele(vct):
    
    hauteur = vct.size()[1]
    vct_traj = conflit.calcul_traj(vct)
    sep = conflit.calcul_crit(vct_traj)
    c = conflit.calcul_cout(vct)
    
    cond_sep = (sep > 0)
    sol = torch.where(cond_sep, 1+sep, c/(0.5+c))
    
    return sol.resize(1,hauteur)


def cercle_avions(n):
    
    liste_avions=[]
    
    for i in range(n):
        
        angle = 2*math.pi*i/N_AVIONS
        x_a = math.cos(angle)
        y_a = math.sin(angle)
        
        a = RAYON*torch.tensor([x_a,y_a])
        b = RAYON*torch.tensor([-x_a,-y_a])
        liste_avions.append(flights.Flight(a, b, DELTA_T, VITESSE))
        
    return liste_avions


if __name__ == '__main__':
    
    tic = time.perf_counter()
    turtle.bgcolor("black")
    
    liste_avions = cercle_avions(N_AVIONS)
    conflit = conflicts.Conflict(liste_avions, NP, SEPARATION)
    diff_evol = algo.DiffEvol(CR, F, NP, NG, RAYON, VITESSE, conflit.N)
    
    vct_para = diff_evol.opti_vect(f_isocele, 3*len(liste_avions), "avion", conflict=True)
    vct_traj = conflit.calcul_traj(vct_para.unsqueeze_(1))
    
    for i in range(N_AVIONS):
        liste_avions[i].trajectoire = vct_traj[0,0,i,0,:,:]
    
    #turtle    
    t = 0
    t_2 = 5*liste_avions[0].pas_de_temps*len(liste_avions) #time?
    while t < liste_avions[0].liste_temps[-1] - t_2:
        
        t = t + t_2
        for avion in liste_avions:
            avion.plot_turtle(t)
    
    turtle.done()

    #plot   
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for avion in liste_avions:
        x,y = avion.plot_graph()
        plt.plot(x, y, 'go')
    
    #cercles dans le plot
    l_t = [i*2*math.pi/999 for i in range(1000)]
    plt.plot([RAYON*math.cos(t) for t in l_t], [RAYON*math.sin(t) for t in l_t], 'w--')
    plt.plot([RAYON*0.66*math.cos(t) for t in l_t], [RAYON*0.66*math.sin(t) for t in l_t], 'w--')
    plt.plot([RAYON*0.33*math.cos(t) for t in l_t], [RAYON*0.33*math.sin(t) for t in l_t], 'w--')
    
    ax.set_aspect('equal','box')
    ax.set_facecolor((0.1,0.1,0.1))
    
    plt.grid()
    plt.show()

    tac = time.perf_counter()
    print("Temps d'éxecution : " + str(tac-tic))
    
    