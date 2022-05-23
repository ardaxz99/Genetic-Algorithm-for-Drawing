import random
import cv2
import numpy as np
from PIL import Image
import sys
import math
from IPython.display import clear_output
import matplotlib.pyplot as plt
from numpy.random import randn
from time import sleep
from skimage.metrics import structural_similarity as ssim
import time

early_stop=True
show_img=False
update_image_per_generation=100
image_save_path='D:\\genetic\\'
source_img_path='D:\\pic.png'
source_img = cv2.imread(source_img_path) 

source_img = cv2.cvtColor(source_img , cv2.COLOR_BGR2RGB)

img_width=180
img_height=180
max_circle_radius=50
num_genes=50

tm_size=20
mutation_type="guided"
mutation_prob=0.2
num_generation=100000

num_inds=20
frac_elites=0.2
frac_parents=0.6
num_elites=int(num_inds*frac_elites)
num_parents=int(num_inds*frac_parents)


fit_array=[] #this array is used to save the best fitness value in the population for all generations

def create_offsprings(elite_pop,parents_after_mutation):
  off=parents_after_mutation+elite_pop
  return off

def init_population(num_inds, num_genes):
  pop = []
  for i in range(0, num_inds):
    chromosome=[]
    for j in range(0, num_genes):
      #radius_len=int(max_circle_radius*random.random())
      radius_len=random.randint(1, max_circle_radius)
      gene= [int((img_width+radius_len/2)*random.random()), int((img_height+radius_len/2)*random.random()), radius_len, 
                                int(0xff*random.random()), int(0xff*random.random()), int(0xff*random.random()), random.random()]
      #gene=[x,y,rad,R,G,B,A]
      chromosome.append(gene)
      
      chromosome =  sorted(chromosome, key=lambda x: x[2], reverse=True)
     
    pop.append(chromosome)
  return pop

def evaluate_fitness(pop,num_inds, num_genes,background,generation):
  imgg = np.zeros(source_img.shape,dtype=np.uint8)
  for j in range(num_inds): 
    
    pop[j].append(0)
    for i in range(num_genes):
      
      overlay = background.copy()
      center=pop[j][i][:2]
      rad=pop[j][i][2]
      rgb=pop[j][i][3:6]
      alpha=pop[j][i][6]
      
      cv2.circle(overlay, tuple(center) , rad, tuple(rgb),-1)
      
      cv2.addWeighted(overlay, alpha, background, 1 - alpha, 0, imgg)
    
    fitness_value=(-1)*np.sum(((imgg/255 - source_img/255)**2))
    
    
    pop[j][num_genes]=fitness_value
  
  population = sorted(pop, key=lambda agent: agent[num_genes], reverse=True)
  canv=draw_canv(population[0],background,generation)
  fit=pop[0][num_genes]
  return canv,fit

def draw_canv(popp,background,generation):
  canvas = np.zeros(source_img.shape,dtype=np.uint8)
  for i in range(num_genes):
    overlay = background.copy()
    center=popp[i][:2]
    rad=popp[i][2]
    rgb=popp[i][3:6]
    alpha=popp[i][6]
      
    cv2.circle(overlay, tuple(center) , rad, tuple(rgb),-1)
      
    cv2.addWeighted(overlay, alpha, background, 1 - alpha, 0, canvas)

  
  fitness_background=(-1)*np.sum(((background/255 - source_img/255)**2))



  if popp[num_genes] > fitness_background:
    fit_array.append(popp[num_genes])
    if generation % update_image_per_generation ==0:

      plt.savefig(image_save_path + "img" + str(generation))
      if show_img==True:
        cv2.imshow('image', canvas)
        cv2.waitKey(1)



    return canvas
  else:
    if generation % update_image_per_generation ==0:
      plt.savefig(image_save_path + "img" + str(generation))
      if show_img == True:
        cv2.imshow('image', canvas)
        cv2.waitKey(1)


    fit_array.append(fitness_background)
    return background




def mutation(parents_after_crossover):
  for i in range(num_parents):   
    for j in range(num_genes):
      if random.random() < mutation_prob:
        if mutation_type=="unguided":
          radius_len=random.randint(1, max_circle_radius)
          parents_after_crossover[i][j][0]=int((img_width+radius_len/2)*random.random())
          parents_after_crossover[i][j][1]=int((img_height+radius_len/2)*random.random())
          parents_after_crossover[i][j][2]=radius_len
          parents_after_crossover[i][j][3]=int(0xff*random.random())
          parents_after_crossover[i][j][4]=int(0xff*random.random())
          parents_after_crossover[i][j][5]=int(0xff*random.random())
          parents_after_crossover[i][j][6]=random.random()
        
        elif mutation_type=="guided":
          radius_len=random.randint(1, max_circle_radius)
          parents_after_crossover[i][j][0]=int(random.uniform(parents_after_crossover[i][j][0]-img_width/8,parents_after_crossover[i][j][0]+img_width/8))
          parents_after_crossover[i][j][1]=int(random.uniform(parents_after_crossover[i][j][1]-img_height/8,parents_after_crossover[i][j][1]+img_height/8))
          parents_after_crossover[i][j][2]=int(random.uniform(parents_after_crossover[i][j][2]-5,parents_after_crossover[i][j][2]+5))
          parents_after_crossover[i][j][3]=int(random.uniform(parents_after_crossover[i][j][3]-32,parents_after_crossover[i][j][3]+32))
          parents_after_crossover[i][j][4]=int(random.uniform(parents_after_crossover[i][j][4]-32,parents_after_crossover[i][j][4]+32))
          parents_after_crossover[i][j][5]=int(random.uniform(parents_after_crossover[i][j][5]-32,parents_after_crossover[i][j][5]+32))
          parents_after_crossover[i][j][6]=random.uniform(parents_after_crossover[i][j][6]-0.25,parents_after_crossover[i][j][6]+0.25)
          
          if parents_after_crossover[i][j][2] > max_circle_radius:
            parents_after_crossover[i][j][2]=max_circle_radius
          if parents_after_crossover[i][j][2] < 0:
            parents_after_crossover[i][j][2]=1         
          if parents_after_crossover[i][j][0] > (img_width+parents_after_crossover[i][j][2]/2):
            parents_after_crossover[i][j][0]=int((img_width+parents_after_crossover[i][j][2]/2))
          if parents_after_crossover[i][j][0] <0 :
            parents_after_crossover[i][j][0]=0        
          if parents_after_crossover[i][j][1] > (img_height+parents_after_crossover[i][j][2]/2):
            parents_after_crossover[i][j][1]=int((img_height+parents_after_crossover[i][j][2]/2))
          if parents_after_crossover[i][j][1] <0 :
            parents_after_crossover[i][j][1]=0  
          if parents_after_crossover[i][j][3] > 255:
            parents_after_crossover[i][j][3]=255
          if parents_after_crossover[i][j][4] > 255:
            parents_after_crossover[i][j][4]=255          
          if parents_after_crossover[i][j][5] > 255:
            parents_after_crossover[i][j][5]=255
          if parents_after_crossover[i][j][3] < 0:
            parents_after_crossover[i][j][3]=0
          if parents_after_crossover[i][j][4] < 0:
            parents_after_crossover[i][j][4]=0
          if parents_after_crossover[i][j][5] < 0:
            parents_after_crossover[i][j][5]=0 
          if parents_after_crossover[i][j][6] < 0:
            parents_after_crossover[i][j][6]=0 
          if parents_after_crossover[i][j][6] > 1:
            parents_after_crossover[i][j][6]=1
  return parents_after_crossover

def crossover(parents):
  i=0
  parents_after_crossover=[]

  if num_parents==1:
    parents_after_crossover=parents
  elif num_parents%2==1:
    while i < num_parents-1:

      p1,p2=parents[i],parents[i+1]
      
      for x in range(num_genes):
        if random.random()>0.5:
          dummy=p1[x]
          p1[x]=p2[x]
          p2[x]=dummy

      
      parents_after_crossover.append(p1)
      parents_after_crossover.append(p2)
      
      i=i+2
    
    p3=parents[num_parents-1]
    parents_after_crossover.append(p3)

  else:
    while i < num_parents:

      p1,p2=parents[i],parents[i+1]
    
      for x in range(num_genes):
        if random.random()>0.5:
          dummy=p1[x]
          p1[x]=p2[x]
          p2[x]=dummy

      
      parents_after_crossover.append(p1)
      parents_after_crossover.append(p2)
      i=i+2
  return parents_after_crossover


def tournament_selection(population):


  population = sorted(population, key=lambda agent: agent[num_genes], reverse=True)
  N=num_inds-num_elites
  popul_without_elites=population[-N:]

  elite_pop=population[0:num_elites]

  parents = random.choices(popul_without_elites, k=tm_size)
  
  parents = sorted(parents, key=lambda agent: agent[num_genes], reverse=True)

  parents=parents[0:num_parents]

  random.shuffle(parents)

  return parents,elite_pop


pop=init_population(num_inds, num_genes)

img = np.zeros(source_img.shape,dtype=np.uint8)
img.fill(128) # or img[:] = 255

canv,fit=evaluate_fitness(pop,num_inds, num_genes,img,generation=0)

if early_stop==True:
  generation=0 # might be commented

  while fit < -100:
    parents,elite_pop=tournament_selection(pop)
    parents_after_crossover=crossover(parents)
    parents_after_mutation=mutation(parents_after_crossover)
    pop=create_offsprings(elite_pop,parents_after_mutation)

    canv,fit=evaluate_fitness(pop,np.shape(pop)[0],num_genes,canv,generation)
    generation=generation+1 # might be commented
    sys.stdout.write("\rfitness value %i Number of generation %i" % (fit, generation))
    sys.stdout.flush()

else:
  for generation in range(num_generation):
    parents,elite_pop=tournament_selection(pop)
    parents_after_crossover=crossover(parents)
    parents_after_mutation=mutation(parents_after_crossover)
    pop=create_offsprings(elite_pop,parents_after_mutation)

    canv,fit=evaluate_fitness(pop,np.shape(pop)[0],num_genes,canv,generation)

    sys.stdout.write("\rfitness value %i Number of generation %i" % (fit, generation))
    sys.stdout.flush()



plt.plot(fit_array)
plt.title("Fitness Plot")        

plt.xlabel("Number of Generations")          
plt.ylabel("Fitness Value")
plt.savefig(image_save_path+"fitness_plot")

