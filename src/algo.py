import torch
import random
import math


class DiffEvol():
    """
    Differential evolution : class that optimizes a problem by progressivily 
    trying to improve a candidate solution with regard to a given measure of quality
    """
    
    def __init__(self, CR, F, NP, NG, r, v, N):
        """
        CR: crossover probability
        F: differential weight (scalaire)
        NP: population number
        NG: generation number
        r,v,N : rayon, vitesse, nombre de points ; à utiliser seulement pour conflits aériens
        """
        self.CR = CR
        self.F = F
        self.NP = NP
        self.NG = NG
        
        self.r = r
        self.v = v
        self.N = N
        
        self.popu = None #matrice population NPxDimension
        self.image = None #image de population
        
        #pour conflits aériens avec main.py
        self.device = torch.device("cpu")
        #pour tester GPU avec main_bench.py
        #self.device = torch.device("gpu" if torch.cuda.is_available() else "cpu")
        
    
    def random_vectors(self):
        """
        Method that chooses 3 random line vectors of population matrix
        """
        l,j,k = random.randint(0,self.NP-1), random.randint(0,self.NP-1), random.randint(0,self.NP-1) 
        a = self.popu[l]
        b = self.popu[j]
        c = self.popu[k]
        return a,b,c
    
    
    def vector_shuffle(self): #a commenter
        """
        Method that shuffles around the lines of population matrix 
        and returns 3 random lines from the new version
        """
        proba_weight = torch.ones((self.NP,self.NP)) - torch.eye(self.NP) 
        lines_ind_rand = torch.multinomial(proba_weight, 3, replacement = False).t() 
        return self.popu[lines_ind_rand[0]], self.popu[lines_ind_rand[1]], self.popu[lines_ind_rand[2]]
    
    
    def flight_domain(self): #a commenter
        """
        Méthode qui crée les domaines des variables nécessaires pour lancer conflits aériens
        """        
        t0_domain = 2*self.r/self.v
        self.time_step = t0_domain/self.N
        
        popu = (torch.rand((self.NP,self.dim))-0.5)*math.pi/3
        ind_0 = torch.arange(self.dim)
        ind_1 = (ind_0 - 1 )% self.dim
        
        cond0 = (ind_0 % 3 == 1)
        t0 = torch.rand((self.NP,self.dim))*t0_domain
        popu = torch.where(cond0, t0, popu)
        
        cond1 = (ind_0 % 3 == 2)
        t0_1 = t0[:,ind_1] #translaté de 1
        t1 = t0_1 + torch.rand((self.NP,self.dim))*(t0_domain-t0_1)
        popu = torch.where(cond1, t1, popu)
        
        return popu
    
    
    def domain_verif(self,popu): #a commenter
        """
        Méthode qui vérifie si les variables pour lancer conflits aériens sont toujours dans les bons domaines
        """
        domain = 2*self.r/self.v
        self.time_step = domain/self.N
        div = self.dim//3
        popu = popu.resize(self.NP*div,self.dim//div)
        
        cond_pos = popu[:,0] > 0.7 #0.7 radians = 40 degrés
        cond_neg = popu[:,0] < 0.7
        popu[cond_pos,0] = 0.7
        popu[cond_neg,0] = 0.7
        
        popu[:,1:] = torch.abs(popu[:,1:])
        popu[:,2] = torch.where(popu[:,2] > 0.5*(domain+popu[:,1]) - self.time_step, (0.5*(domain+popu[:,1])-2*self.time_step) * torch.ones(self.NP*div), popu[:,2])
        popu[:,1] = torch.where(popu[:,1] > popu[:,2] - self.time_step, popu[:,2]-2*self.time_step, popu[:,1])
        popu[:,1] = torch.where(popu[:,1] == 0, popu[:,1]+ 2*self.time_step, popu[:,1])
        popu = popu.resize(self.NP,self.dim)
        
        return popu
    
    
    def opti_iter(self, function, dim, domain, benchmark=False):
        """
        Optimizes a given function of dimension n iteratively inside a chosen domain
        Returns individual closest to minimum point
        """

        self.popu = (torch.rand((self.NP,dim))-0.5)*(2*domain)
        self.image = torch.tensor([function(self.popu[i,:]) for i in range(self.NP)])
        
        for gen in range(self.NG):
        
            temp_popu = torch.zeros((self.NP,dim)) #temporary population 
            
            for i in range(self.NP):

                x = self.popu[i] 
                a,b,c = self.random_vectors()
                
                y = a + self.F*(b-c) 
                
                x_new = torch.where((torch.rand(dim)<self.CR), y, x)
                
                rand_ind = random.randint(0,dim-1) 
                x_new[rand_ind] = y[rand_ind]
                
                condition = function(x_new) < function(x)
                x = torch.where(condition, x_new, x)
                temp_popu[i,:] = x
                
            self.popu = temp_popu #population update
            self.image = torch.tensor([function(self.popu[i,:]) for i in range(self.NP)]) #image update
        
        min_ind_image = torch.argmin(self.image)
        
        if benchmark : return "Minimum : " + str(self.popu[min_ind_image]) + " ; " + "Image : " + str(self.image[min_ind_image])
        else : return self.popu[min_ind_image]
    
    
    def opti_vect(self, function, dim, domain, benchmark=False, conflict=False): #a commenter
        """
        Optimizes a given function of dimension n inside a chosen domain faster and better than by iteration
        Returns individual closest to minimum point
        """
        self.dim = dim
        self.domain = domain
        
        if self.domain == "avion" : self.popu = self.flight_domain()
        else : self.popu = (torch.rand((self.NP,dim))-0.5)*(2*self.domain)
        
        self.popu = self.popu.to(self.device)
        self.image = function(self.popu.t()).to(self.device)
        
        for gen in range(self.NG):
            
            if conflict : print(gen+1)
            
            a,b,c = self.vector_shuffle()
            y = a + self.F*(b-c)
            
            rand_ind = torch.randint(0, dim, (self.NP,), dtype = torch.long, device=self.device)
            component_condition = torch.eye(dim)[rand_ind].byte().to(self.device)
            new_popu=torch.where((torch.rand((self.NP,dim), device=self.device)<self.CR) | component_condition, y, self.popu)
            
            if conflict : new_popu = self.domain_verif(new_popu) 
            
            image_new_popu = function(new_popu.t())
            condition = image_new_popu < self.image
            self.popu = torch.where(condition, new_popu.t(), self.popu.t()).t()
            self.image = torch.where(condition, image_new_popu, self.image)  
        
        min_ind_image = torch.argmin(self.image)
        
        if benchmark : return "Minimum : " + str(self.popu[min_ind_image]) + " ; " + "Image : " + str(self.image[min_ind_image])
        else : return self.popu[min_ind_image]

