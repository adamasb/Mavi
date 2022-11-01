# mavi

## Main entry points:

Fairly nasty environment-class (visualization could really use a cleanup) in 
 - `mazeenv/maze_environment.py`

Example 'agent' which computes a randomized value function and visualize it:
 - `basic/viagent.py`

Course exercises (very little of which is needed)
 - `irlc`

## Tasks + plan
```
pip install gym==0.21.0
```

 - Foerst: Faa almindelig value-iteration til at fungere i Mazebase-gridworld. Dvs. vi antager at vi kender dynamikken p_ij og r_ij, r_ij^out, og planlaegger paa en horizont paa K. Dvs. vi vaelger phi-funktionen i VP paperet og implementerer algoritmen i numpy. Den skal kunne planlaegge at gaa til et maal optimalt. 
  - Implementer en paralell torch-version af koden og check at de to implementationer giver samme output. Dette check kan senere bruges til at lave en effektiv torch-version. 
  - Actor-critic (dvs. foerste algoritme i section 4). Check at vi kan navigere maze-environmentet. 
  - Multiagents: Speaker-listener opgaven (simpleste ikke-trivielle)
  - Actor-critic baseline (samme implementation som actor critic men v(s) er et almindeligt 2-lags NN)
  
