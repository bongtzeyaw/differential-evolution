import torch


class Conflict():
    
    def __init__(self,liste_avions,NP,sep):
        
        self.NP = NP
        self.sep = sep
        self.nb_avions = len(liste_avions)
        self.vols = liste_avions
        
        self.points = torch.zeros(self.nb_avions)
        for i,vol in enumerate(self.vols):
            self.points[i] = vol.distance/(vol.delta_t*vol.v)
        self.N = int(torch.max(self.points))
        
        self.t_max = liste_avions[0].liste_temps[-1]
        self.t_trajectoires = torch.zeros(int(NP), 1, self.nb_avions, 1, self.N, 2) #vecteur contenant les trajectoires
        self.t_temps = (torch.zeros(1, 1, self.nb_avions, 1, self.N) + self.vols[0].liste_temps).unsqueeze_(5)
        self.t_pas_de_temps = torch.zeros(self.nb_avions)
        
        self.t_cap = torch.zeros(1,1,self.nb_avions,1,1,1)
        self.h = torch.zeros(1,self.nb_avions,1,1,1)
        self.orig = torch.zeros(1,1,self.nb_avions,1,1,2)
        self.dest = torch.zeros(1,1,self.nb_avions,1,1,2)
        self.u_orig = torch.zeros(1,1,self.nb_avions,1,1,2)
        
        for i,vol in enumerate(self.vols):
            points_manquants = int(self.N - self.points[i])
            if points_manquants != 0:
                fin_traj = torch.ones(points_manquants)*vol.dest
                vol.trajectoire = torch.cat((vol.trajectoire,fin_traj),0)
        
        for i in range(self.nb_avions):
            self.t_trajectoires[:,:,i,:,:,:] = self.vols[i].trajectoire_init
            self.t_pas_de_temps[i] = (self.vols[i].liste_temps[1] - self.vols[i].liste_temps[0])
            self.t_cap[:,:,i,:,:,:] = self.vols[i].cap
            self.h[:,i,:,:,:] = self.vols[i].delta_d
            self.orig[:,:,i,:,:,:] = self.vols[i].orig
            self.dest[:,:,i,:,:,:] = self.vols[i].dest
            self.u_orig[:,:,i,:,:,:] = self.vols[i].u_clone
        
        self.t_trajectoires_init = torch.clone(self.t_trajectoires)
        self.range = torch.arange(self.N).type(torch.FloatTensor) 
        for _ in range(4):
            self.range = self.range.unsqueeze_(0)  
        self.range = self.range.unsqueeze_(5)
        
        self.liste_t0 = torch.zeros(self.NP,1,self.nb_avions,1,1,1)
        self.liste_t1 = torch.zeros(self.NP,1,self.nb_avions,1,1,1)
        self.liste_tf = torch.zeros(self.NP,1,self.nb_avions,1,1,1)
        self.alpha = torch.zeros(self.NP,1,self.nb_avions,1,1,1)
        
        self.v = self.vols[0].v
        self.v_sur_h = self.v/self.h
        
        self.normal_orig = torch.zeros(1,1,self.nb_avions,1,1,2) #creation du vecteur normal
        self.normal_orig[0,0,:,0,0,0] = -self.u_orig[0,0,:,0,0,1]
        self.normal_orig[0,0,:,0,0,1] = self.u_orig[0,0,:,0,0,0]

        self.identité = torch.zeros(NP,1,self.nb_avions,self.nb_avions,self.N)
        for i in range(self.nb_avions):
            self.identité[:,:,i,i,:] = 1.
        
        self.ind = torch.arange(self.nb_avions*3)
        self.h = self.h.unsqueeze_(0)
        self.zeros = torch.zeros(self.NP,1,self.nb_avions,self.nb_avions,self.N)
        
        
    def calcul_crit(self,vct_traj):
        
        matrice_distance = vct_traj - torch.transpose(vct_traj,2,3)
        distance = matrice_distance[:,:,:,:,:,0]**2 + matrice_distance[:,:,:,:,:,1]**2
        distance = self.sep**2 - distance - self.identité*self.sep**2
        
        cond = 0 < distance
        distance = torch.where(cond, distance, self.zeros)
        somme = torch.sum(distance,(2,3,4))/2
        
        return somme
        
    
    def calcul_traj(self,vct_para):
        
        self.t_trajectoires = self.t_trajectoires_init
        pas_de_temps = self.t_pas_de_temps[0]
        
        x = torch.clone(vct_para.t())
        self.liste_t1[:,0,:,0,0,0] = x[:,self.ind%3 == 2]
        self.liste_t0[:,0,:,0,0,0] = x[:,self.ind%3 == 1]
        self.alpha[:,0,:,0,0,0] = x[:,self.ind%3 == 0]
        self.liste_tf = 2*self.liste_t1-self.liste_t0

        #Calcul des points
        T_N = self.v*(self.liste_t1-self.liste_t0)/self.h + 1 
        T_n= self.v*(self.liste_t1-self.liste_t0)*torch.cos(self.alpha)/self.h 
        Nmax = int(torch.max(T_N))
        nmax = int(torch.max(T_n))
      
        #Calcul de nouv temps
        self.range = torch.arange(self.N + Nmax - nmax).type(torch.FloatTensor) 
        for _ in range(4):
            self.range = self.range.unsqueeze_(0)
        self.range = self.range.unsqueeze_(5)
        
        self.T_temps_1 = torch.zeros(1,1,self.nb_avions,1,self.N + Nmax - nmax) + self.range[:,0,:,:,:,0]*self.h[:,0,:,:,:,0]/self.v
        self.T_temps_1 = self.T_temps_1.unsqueeze_(5)
       
        #création fin
        FIN = torch.ones(self.NP,1,self.nb_avions,1,Nmax-nmax,2)*self.dest
        self.t_trajectoires = torch.cat((self.t_trajectoires,FIN),4)
        self.t_trajectoires_ini_bis = torch.cat((self.t_trajectoires_init,FIN),4)

        #Création points triangle                             
        u_v_div_h = self.u_orig*self.v_sur_h 
        Pos_T0 = self.orig + u_v_div_h*self.liste_t0
        Pos_T1 = Pos_T0 + (self.u_orig + self.normal_orig*torch.tan(self.alpha))*self.v_sur_h*(self.liste_t1-self.liste_t0)*torch.cos(self.alpha)
        Pos_Tf = self.orig + u_v_div_h*(self.liste_t0 + 2*(self.liste_t1-self.liste_t0)*torch.cos(self.alpha))

        #création du premier chemin dévié
        cond0 =  (self.T_temps_1 < self.liste_t1 + (T_N - T_n)*pas_de_temps)*(self.T_temps_1 > self.liste_t0 )
        u = (Pos_T1 - Pos_T0)/(T_N-1)
        chemin0 = Pos_T0 + u * (self.range - self.liste_t0*self.v_sur_h)
        self.t_trajectoires = torch.where(cond0,chemin0,self.t_trajectoires)        

        #création du retour
        cond1 = ((self.T_temps_1 < self.liste_t1 + (2*T_N-T_n)*pas_de_temps) * (self.T_temps_1 > self.liste_t0 + T_N*pas_de_temps ))
        u =(Pos_Tf - Pos_T1)/(T_N-1)
        chemin1 = Pos_T1 + u *(self.range - self.liste_t1*self.v_sur_h)
        self.t_trajectoires = torch.where(cond1,chemin1,self.t_trajectoires)
        
        #création fin
        cond_suite = (self.T_temps_1 > self.liste_tf)
        chemin_fin = Pos_Tf + self.u_orig*(self.range - self.liste_tf *self.v_sur_h)
        self.t_trajectoires = torch.where(cond_suite,chemin_fin,self.t_trajectoires)
        
        #condition ligne droite
        cond_fin =  (self.liste_t0 > 0) *(self.liste_t0 < self.liste_t1 - pas_de_temps)*( self.liste_t1 <= 0.5*(self.t_max + self.liste_t0))
        t_trajectoires = torch.where(cond_fin,self.t_trajectoires,self.t_trajectoires_ini_bis)
        
        #Quelques changements ... 
        self.identité = torch.zeros(self.NP,1,self.nb_avions,self.nb_avions,self.N+ Nmax - nmax)
        for i in range(self.nb_avions):
            self.identité[:,:,i,i,:] = 1.
        self.zeros = torch.zeros(self.NP,1,self.nb_avions,self.nb_avions,self.N+ Nmax - nmax)
            
        return t_trajectoires
    

    def calcul_cout(self,vct_para):
        """
        Calcul du cout modelisé par l'aire du triangle
        """
        size = vct_para.size()
        x = torch.clone(vct_para.t())
        div = size[0]//3
        x = x.resize(size[1]*div,size[0]//div)
        
        surf = ((x[:,2] - x[:,1])**2)*(self.v**2)*torch.cos(x[:,0])*torch.sin(x[:,0])
        surf = surf.resize(div,size[1])     
        
        return torch.sum(surf**2,-2).resize(size[1],1)#/ air_disque**2
    
