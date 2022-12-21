from scipy.spatial import KDTree
import random
import numpy as np
from operator import itemgetter

class NovArchive:
    """Archive used to compute novelty scores."""
    def __init__(self, lbd, lfitness, k=15):
        self.all_bd=lbd
        self.kdtree=KDTree(self.all_bd)
        self.k=k
        self.list_fitness = lfitness
        #print("Archive constructor. size = %d"%(len(self.all_bd)))
        
        
    def update(self,new_bd,new_fitness):
        oldsize=len(self.all_bd)
        self.all_bd=self.all_bd + new_bd
        self.kdtree=KDTree(self.all_bd)
        self.list_fitness += new_fitness
        # print("Archive updated, old size = %d, new size = %d"%(oldsize,len(self.all_bd)))
        
        
    def get_nov(self,ind, population=[]):
        bd = ind.bd
        # print("passage get_nov")
        # print("bd : ",bd)
        # print("population : ",population)
        # input()
        dpop=[]
        for ind in population:
            dpop.append((np.linalg.norm(np.array(bd)-np.array(ind.bd)),np.std(ind)))
        darch,ind=self.kdtree.query(np.array(bd),self.k)
        d=dpop+list(darch)
        
        d=dpop + [(darch[i],self.list_fitness[ind[i]]) for i in range(len(darch))]
        
        d.sort(key = lambda e:e[0])
        
        d,std = list(map(list,zip(*d)))
        
        if (d[0]!=0):
            print("WARNING in novelty search: the smallest distance should be 0 (distance to itself). If you see it, you probably try to get the novelty with respect to a population your indiv is not in. The novelty value is then the sum of the distance to the k+1 nearest divided by k.")
        novelty = sum(d[:self.k+1]) / self.k
        
        # Ajout TME competition_objective
        
        competition_objective = len([i for i in std[1:self.k +1] if i<std[0]]) # sum([1 for x in self.list_fitness[:self.k+1] if x < self.list_fitness[self.k+1]])
        return novelty, competition_objective
        


    
    
    
    
    

    def size(self):
        return len(self.all_bd)
    
def updateNovelty(population, offspring, archive, k=15, add_strategy="random", _lambda=6, verbose=False):
    """Update the novelty criterion (including archive update) 
   Implementation of novelty search following (Gomes, J., Mariano, P., & Christensen, A. L. (2015, July). Devising effective novelty search algorithms: A comprehensive empirical study. In Proceedings of GECCO 2015 (pp. 943-950). ACM.).
   :param population: is the set of indiv for which novelty needs to be computed
   :param offspring: is the set of new individuals that need to be taken into account to update the archive (may be the same as population, but it may also be different as population may contain the set of parents)
   :param k: is the number of nearest neighbors taken into account
   :param add_strategy: is either "random" (a random set of indiv is added to the archive) or "novel" (only the most novel individuals are added to the archive).
   :param _lambda: is the number of individuals added to the archive for each generation
   The default values correspond to the one giving the better results in the above mentionned paper.
   The function returns the new archive
   """
    weight1 = 1.0
    weight2 = 1.0
    

   # Novelty scores updates
    if (archive) and (archive.size()>=k):
        if (verbose):
            print("Update Novelty. Archive size=%d"%(archive.size())) 
        for ind in population:
            novelty,competition_objective = archive.get_nov(ind, population)
            ind.competition_local = competition_objective
            ind.novelty = novelty
           #print("Novelty: "+str(ind.novelty))
    else:
        if (verbose):
            print("Update Novelty. Initial step...") 
        for ind in population:
            ind.novelty=0.
            ind.competition_local=0.

    if (verbose):
        print("Fitness (novelty): ",end="") 
        for ind in population:
            print("%.2f, "%(ind.novelty),end="")
        print("")
    if (len(offspring)<_lambda):
        print("ERROR: updateNovelty, lambda(%d)<offspring size (%d)"%(_lambda, len(offspring)))
        return None

    lbd=[]
    lfitness=[]
   # Update of the archive
    if(add_strategy=="random"):
       # random individuals are added
        l=list(range(len(offspring)))
        random.shuffle(l)
        if (verbose):
            print("Random archive update. Adding offspring: "+str(l[:_lambda])) 
        lbd=[offspring[l[i]].bd for i in range(_lambda)]
        lfitness=[np.std(offspring[l[i]]) for i in range(_lambda)]
        
        
        
    elif(add_strategy=="novel"):
       # the most novel individuals are added
        soff=sorted(offspring,lambda x:x.novelty)
        ilast=len(offspring)-_lambda
        lbd=[soff[i].bd for i in range(ilast,len(soff))]
        lfitness=[np.std(soff[i]) for i in range(ilast,len(soff))]
        if (verbose):
            print("Novel archive update. Adding offspring: ")
            for offs in soff[iLast:len(soff)]:
                print("    nov="+str(offs.novelty)+" fit="+str(offs.fitness.values[0])+" bd="+str(offs.bd))
    else:
        print("ERROR: updateNovelty: unknown add strategy(%s), valid alternatives are \"random\" and \"novel\""%(add_strategy))
        return None
       
    if(archive==None):
        archive=NovArchive(lbd,lfitness,k)
    else:
        archive.update(lbd,lfitness)

    return archive