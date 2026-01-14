This e-mail was attached to the data.

Dear Silvia,
here is some material for your student, as promised. A few indications:
1) The code is now tuned for a Monte Carlo estimation of transport
    properties of O_{2}. The main program is called Omega22_O2.f90.
    For you, the interesting routine is Collision(v1,omega1,v2,omega2,2,2) (located in Sources/Sorgenti_cambiate/CT_Collider.f90)
   accepting initial linear and angular velocities of collision pairs and restituting
   the same variables after a trajectory calculation. The last two integers identify
   the colliding species: 1 is Nitrogen, 2 is Oxygen.

2) Since I never learnt how to write a make file, the present code exe file is produced
    in the folder Exe_file by invoking the script build_exe.sh in the folder Compilation.
    In its present form, the Intel compiler is used (ifx). You can replace with any other
    available.

3) The main program is a serial one but you can turn it into a parallel one very easily
     by OpenMP, since trajectories are independent.

4) Some trajectories go close to some problematic configurations which can result
    in more or less large violations of energy conservation. In this case, the program
    outputs a warning in a file. 

5) For any problem, you know where to find me.

Have a nice week end

Aldo
