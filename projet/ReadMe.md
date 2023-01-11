# Intelligence Artificielle pour la Robotique

## Projet : Reinforcement Learning in Continuous Action Spaces

Vous trouverez dans ce repo le code et les notebooks que nous avons produit dans le but de reproduire les résultats du papier
Reinforcement Learning in Continuous Action Spaces (2007) de Hado van Hasselt et Marco A. Wiering qui présente une famille
d'algorithme d'apprentissage par renforcement appelée **CACLA (Continuous Actor Critic Learning Automaton)**.
Nous avons aussi été explorer d'autre piste comme le passage par **batch**, la mise à jour commune de l'acteur et du critique lorsque
l'erreur de différence temporelle est positive ou encore la comparaison avec des algorithmes de Deep RL plus récent tel que 
**PPO** / **DDPG**.

<hr>

### Algorithmes implémentés :

Nous avons implémentés les trois versions de la famille CACLA à savoir **CAC**, **CACLA** et **CACLA+VAR**. Ils sont disponibles dans les fichiers suivants :
- `./utils/CAC.py`
- `./utils/CACLA.py`
- `./utils/CACLAVAR.py`

Les versions par batch :

- `./utils/CACLA_batch.py`
- `./utils/CACLAVAR_batch.py`

L'Acteur et le Critique ont été implémenté dans des fichiers séparés ici :
- `./utils/Actor.py`
- `./utils/Critic.py`

<hr>

### Environnements implémentés :

Nous avons implémentés les deux environnements cités dans le papier à savoir **Tracking** et **CartPoleContinous** [1] que vous pourrez retrouver ici :

- `./utils/Tracking.py`
- `./utils/CartPoleContinuous.py`

<hr>

### Expérimentations

Nous avons conduit une campagne d'expérimentation aussi proche que possible que celle du papier. Le notebook correspondant au test de **Tracking** ce trouve ici : 
- `./tests/CAC_Global_testing_Tracking.ipynb`

Le notebook correspondant au test sur **CartPoleContinuous** se trouve ici :

- `./tests/CAC_Global_testing_CartPoleContinuous.ipynb`

Le notebook de test comparatif des versions **batchs** se trouve ici :

- `./tests/CACLA_BATCH.ipynb`

Pour tester et reproduire nos résultats des algorithmes de Deep RL comme **DDPG**[2] et **PPO**[3] il faudra executer les fichiers suivants :

- `./utils/extern/PPO_with_tderror.py`
- `./utils/extern/PPO_without_tderror.py`
- `./utils/extern/DDPG_with_tderror.py`
- `./utils/extern/DDPG_without_tderror.py`

Enfin, le notebook comparatif entre **CACLA+VAR** et **PPO** sur LunarLanderContinuous :

- `./tests/CACLA_vs_PPO_lunarlandercontinuous.ipynb`

<hr>

## Source :

- [1] : lien vers l'environnement original que l'on à adapté à nos besoins : https://gist.github.com/iandanforth/e3ffb67cf3623153e968f2afdfb01dc8 

- [2] : lien vers la source : https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch/tree/master/Char05%20DDPG

- [3] : lien vers la source : https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch/tree/master/Char07%20PPO