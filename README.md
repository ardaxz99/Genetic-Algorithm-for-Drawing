# Genetic-Algorithm-for-Drawing

<p align="center">
  <img 
    width="300"
    height="300"
    src="https://user-images.githubusercontent.com/87468948/169909057-f7b19021-7323-43b2-bab8-7c7ee6ffd5ee.png"
  >
</p>

<p align="center">
  <img 
    width="300"
    height="300"
    src="https://user-images.githubusercontent.com/87468948/169909083-a2ac7050-d9e4-4798-b4fa-f30e07b32cc5.png"
  >
</p>



## What it does?  

The computer iteratively attempts to mimic an input pictures through drawing it with circles of varying radii and colours, and continuously improve its accuracy based on a generic genetic algorithm.  

## How it works?  

1-Initialize a population.  
2-Evaluates the fitness of the population.  
3-Select some individuals as elite individuals and directly pass them to the next generation. Parents are selected among the other individuals.  
4-Tournament selection method is preferred.  
5-Cross over and mutation are applied.  
6-After cross-over and mutation, the offspring population is formed.  
7-The fitness of the new population is evaluated, and the same process starts again.  

## The parameters in the python file  

**early_stop** = Decide whether the algorithm stops automatically when the fitness of the population exceeds a treshold value.  
**show_img** = Decide whether to show images of best individual in the population for each generation.  
**update_image_per_generation** = Update shown image for each "update_image_per_generation" number of generation.  
**image_save_path** = Path of the saved/created images. The corresponding image of the best individual in the population at every 1000th generation is created.  
The fitness plot is also generated after "num_generation" step.
**source_img_path** = Path of the original image.  
**max_circle_radius** = Maximum radius of circles  
**num_genes** = Number of circles for each individual  
**tm_size** = Tournament size  
**mutation_type** = Decide whether apply random mutations or guided mutations  
**mutation_prob** = Mutation probability  
**num_generation** = It is similar to the epoch number in neural networks. The algorithm stops after reaching this value.  
**num_inds** = Number of individuals in the population  
**frac_elites** = Fraction of elite individuals in the population  
**frac_parents** = Fraction of selected parents for cross-over and mutation in the population  

## Challenges we ran into  

The most difficult component was attempting to improve our fitness function, cross-breeding function, and mutation function, as the simulation takes a long time and it is quite difficult to obtain substantial results in an hour. As a result, we had to be quite cautious about when we tested.  
We were unable to obtain a clearly optimum answer in the end, but we feel that with future refinements, the method should be able to yield improved results.  

## How to run it  

Change image paths, set the parameters and you are good to go.  
