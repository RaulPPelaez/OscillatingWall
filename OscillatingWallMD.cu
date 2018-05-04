/*Raul P. Pelaez 2018. 
An oscillating wall in a DPD fluid.

The wall is located at the bottom of the box, covering it entirely. The system is periodic in XY, but DPD particles are confined to move inside a certain range in the Z direction (from the top of the wall to a threshold below the top of the box).


The top wall is a reflective one, particles in or above it have their z velocity forced to be negative.

The bottom wall  oscilates in the x direction with a sinusoidal force.

All parameters are read from a data.main (You should have received one with this file). Refer to the function readParameters to see the list of available parameters.


Besides positions and velocities of the particles, the program will output an histogram of the DPD particles x velocities in the z direction.


The code expects a file with the wall particles positions, one with its particle-particle bonds and another with its fixed point bonds.

The wall should have the same size as the box. See genWall.bash


The wall will start to oscilate only after the relaxation steps.
 */

//This include contains the basic needs for an uammd project
#include"uammd.cuh"
//The rest can be included depending on the used modules
#include"Integrator/VerletNVT.cuh"
#include"Interactor/NeighbourList/CellList.cuh"
#include"Interactor/PairForces.cuh"
#include"Interactor/ExternalForces.cuh"
#include"Interactor/BondedForces.cuh"
#include"Interactor/Potential/Potential.cuh"
#include"Interactor/Potential/DPD.cuh"
#include"utils/InitialConditions.cuh"
#include"utils/InputFile.h"
#include<fstream>

double epsilonWall = 5.0;

using namespace uammd;
//This external force takes care of making the wall oscilate
struct WallOscilation: public ParameterUpdatable{
  real w;
  real A;
  real t;
  real relaxTime = -1;
  WallOscilation(real w, real A):w(w),A(A){t=0;}
  __device__ real3 force(const real4 &pos){    
    real3 f = make_real3(A*sinf(w*t),0, 0);
    return f;
  }
  std::tuple<const real4 *> getArrays(ParticleData *pd){
    auto pos = pd->getPos(access::location::gpu, access::mode::read);
    return std::make_tuple(pos.raw());
  }
  
  void updateSimulationTime(real time){
    if(relaxTime<0) relaxTime = time;
      t=time-relaxTime;
  }
    
};


//This kernel reflects the velocity of the particles above the top wall.
template<class GroupIndexIterator>
__global__ void reflectVelocity(GroupIndexIterator groupIndex, real4* pos, real3 *vel, real max_z, int N){
  int id = blockIdx.x*blockDim.x + threadIdx.x;
  if(id>=N) return;
  int index = groupIndex[id];
  if(pos[index].z>=max_z){
    if(vel[index].z > 0) vel[index].z *=real(-1.0);
  }  
}
//Same as Harmonic, but the z position is shifted. This is used for the wall fixed bonds, allowing to displace the wall in the z direction.
struct HarmonicZ: public BondedType::Harmonic{
  real offset;
  HarmonicZ(real offset): offset(offset){}
  inline __device__ real3 force(int i, int j, const real3 &r12, const BondInfo &bi){
    real3 r12z = make_real3(r12.x, r12.y, r12.z-offset);
    return Harmonic::force(i, j, r12z, bi);
  }    
};

using namespace std;

//Parameters
int numberParticlesDPD;
real cutOffWall_fluid;  //cut off between particles in the wall and the fluid (also wall-wall)
real cutOffDPD;         //cut off between fluid particles
real temperature;
real gammaDPD;
real intensityDPD;

real3 boxSize;
real topWall_z;         //From 0 to Lz, Particles wont be allowed above this height

real wallParticleMass;

real wallOscilationAmplitude;
real wallOscilationWaveNumber;

int relaxSteps, numberSteps, printSteps;
real dt;

int nbinsHisto = 100;

std::string wallCoordinatesFile, wallBondsFile, wallBondsFileFP;

std::string outputName;


void readParameters(std::string datamain, shared_ptr<System> sys);

int main(int argc, char *argv[]){
  auto sys = make_shared<System>(argc, argv);
  readParameters("data.main", sys);
  
  int numberParticlesWall;
  ifstream inWall(wallCoordinatesFile);
  inWall>>numberParticlesWall;

  int N = numberParticlesDPD + numberParticlesWall;  
  
  ullint seed = 0xf31337Bada55D00dULL^time(NULL);
  sys->rng().setSeed(seed);

  auto pd = make_shared<ParticleData>(N, sys);

  real wallWidth;
  Box box(boxSize);//({40, 40, 90});
  real densityDPD = numberParticlesDPD/box.getVolume();

  real max_z = topWall_z-boxSize.z/2;//box.boxSize.z/2-2*cutOff;
  real min_z;
  real zOffsetFixedPoint; //Needed due to the wall being moved to the bottom of the box
  //Initial positions
  {
    //Ask pd for a property like so:
    auto pos = pd->getPos(access::location::cpu, access::mode::write);
    auto vel = pd->getVel(access::location::cpu, access::mode::write);
    auto mass = pd->getMass(access::location::cpu, access::mode::write);    
    real mean_wall_z = 0;
    real wallMinz=10000, wallMaxz=-10000;

    fori(0, numberParticlesWall){
      inWall>>pos.raw()[i].x>>pos.raw()[i].y>>pos.raw()[i].z>>pos.raw()[i].w;
      real z = pos.raw()[i].z;
      mean_wall_z += z;

      wallMinz = min(z, wallMinz);
      wallMaxz = max(z, wallMaxz);
      
      pos.raw()[i].w = 1;
      mass.raw()[i] = wallParticleMass;
      vel.raw()[i] = make_real3(sys->rng().gaussian(0, 0.01),
				sys->rng().gaussian(0, 0.01),
				sys->rng().gaussian(0, 0.01));
    }

    wallWidth = wallMaxz - wallMinz;
    min_z = -10000000;
    mean_wall_z /= numberParticlesWall;
    
    fori(0, numberParticlesWall){
      pos.raw()[i].z += -mean_wall_z -box.boxSize.z*0.5 + wallWidth;      
      min_z = max(pos.raw()[i].z, min_z);
    }
    
    min_z += 2*pow(2, 1/6.);

    zOffsetFixedPoint = -mean_wall_z -box.boxSize.z*0.5 + wallWidth;
    //Start in a random configuration
    real3 vcm = make_real3(0);
    auto initial =  initLattice(make_real3(boxSize.x, boxSize.y, max_z-min_z), N, fcc);    
    fori(numberParticlesWall, numberParticlesWall + numberParticlesDPD){
      pos.raw()[i] = make_real4(initial[i-numberParticlesWall].x,
				initial[i-numberParticlesWall].y,
				initial[i-numberParticlesWall].z+min_z+(max_z-min_z)*0.5,
				0);      
      pos.raw()[i].w = 0;
      
      vel.raw()[i] = make_real3(sys->rng().gaussian(0, sqrt(3*temperature)),
				sys->rng().gaussian(0, sqrt(3*temperature)),
				sys->rng().gaussian(0, sqrt(3*temperature)));

      vcm += vel.raw()[i];
      mass.raw()[i] = 1;
    }
    //Remove center of mass velocity
    vcm /= (real) numberParticlesDPD;
    fori(numberParticlesWall, numberParticlesWall + numberParticlesDPD){
      vel.raw()[i] -= vcm;
    }
  }

  auto all_group = make_shared<ParticleGroup>(pd, sys, "All");
  auto Wall_group = make_shared<ParticleGroup>(particle_selector::Type(1), pd, sys, "Wall_particles");
  auto DPD_group = make_shared<ParticleGroup>(particle_selector::Type(0), pd, sys, "DPD_particles");

  
  using NVT = VerletNVT::GronbechJensen;
  
  NVT::Parameters par;
  par.temperature = temperature;
  par.dt = dt;
  par.viscosity = gammaDPD;

  auto verlet = make_shared<NVT>(pd, all_group, sys, par);
  real viscosityDPD = 0.17629/0.3 +
    0.062692*(exp(4.095577*densityDPD)-1) -
    8.743269e-6*(exp(11.124920*densityDPD-1)) +
    2.542477e-6*temperature*(exp(14.863984*densityDPD)-1);
  {//LJ fluid
    using PairForces = PairForces<Potential::LJ>;
    
    auto pot = make_shared<Potential::LJ>(sys);

    Potential::LJ::InputPairParameters par;
    par.epsilon = 1.0;
    par.shift = false;

    par.sigma = 1;
    par.cutOff = 2.5*par.sigma;
    //Once the InputPairParameters has been filled accordingly for a given pair of types,
    //a potential can be informed like this:
    pot->setPotParameters(0, 0, par);
    pot->setPotParameters(1, 1, par);
    par.epsilon = epsilonWall;
    pot->setPotParameters(0, 1, par);


    PairForces::Parameters params;
    params.box = box;  //Box to work on
    auto pairforces = make_shared<PairForces>(pd, all_group, sys, params, pot);
    verlet->addInteractor(pairforces);
  }

  {//Wall bonds
    using BondedForces = BondedForces<BondedType::HarmonicPBC>;
    //You can use Elastic_Network_Model.cpp to generate some example bonds for the starting configuration.
    BondedForces::Parameters params;
    params.file = wallBondsFile.c_str();  //Box to work on
    auto bondedforces = make_shared<BondedForces>(pd, sys, params,
						  BondedType::HarmonicPBC(box));
    verlet->addInteractor(bondedforces);
  }
  {//Wall bonds
    using BondedForces = BondedForces<HarmonicZ>;
    //You can use Elastic_Network_Model.cpp to generate some example bonds for the starting configuration.
    BondedForces::Parameters params;
    params.file = wallBondsFileFP.c_str();  //Box to work on
    auto bondedforces = make_shared<BondedForces>(pd, sys, params,
						  HarmonicZ(zOffsetFixedPoint));
    verlet->addInteractor(bondedforces);
  }

  real wallDensity = numberParticlesWall*wallParticleMass/(box.boxSize.y*box.boxSize.x);
  sys->log<System::MESSAGE>("viscosity: %f", viscosityDPD);
  sys->log<System::MESSAGE>("C: %f", viscosityDPD*densityDPD/(2*M_PI*wallOscilationWaveNumber*pow(wallDensity,2)));
  sys->log<System::MESSAGE>("Penetration length: %f", sqrt(2*viscosityDPD/(2*M_PI*wallOscilationWaveNumber*densityDPD)));
							      

  sys->log<System::MESSAGE>("RUNNING!!!");

  pd->sortParticles();

  ofstream posOut(outputName+".pos");
  ofstream velOut(outputName+".vel");
  ofstream histoOut(outputName+".histo");

  Timer tim;
  tim.tic();
  //Thermalization
  forj(0,relaxSteps){
    {
      auto vel = pd->getVel(access::location::gpu, access::mode::readwrite);
      auto pos = pd->getPos(access::location::gpu, access::mode::read);

      int N_dpd = DPD_group->getNumberParticles();
      int Nthreads = 512;
      int Nblocks  = N_dpd/Nthreads+1;
      reflectVelocity<<<Nblocks, Nthreads>>>(DPD_group->getIndexIterator(access::location::gpu),
					     pos.raw(),
					     vel.raw(),
					     max_z,
					     N_dpd);
    }

    verlet->forwardTime();
  }
  
  cudaDeviceSynchronize();
  //Wall Oscilation
  real A = wallOscilationAmplitude;
  real w = 2*M_PI*wallOscilationWaveNumber;
  auto wallOscilation = make_shared<ExternalForces<WallOscilation>>(pd, Wall_group, sys,
								    make_shared<WallOscilation>(w,A));
  verlet->addInteractor(wallOscilation);
  

  //Run the simulation
  forj(0,numberSteps){
    //This will instruct the integrator to take the simulation to the next time step,
    //whatever that may mean for the particular integrator (i.e compute forces and update positions once)
    {
      auto vel = pd->getVel(access::location::gpu, access::mode::readwrite);
      auto pos = pd->getPos(access::location::gpu, access::mode::read);

      int N_dpd = DPD_group->getNumberParticles();
      int Nthreads = 512;
      int Nblocks  = N_dpd/Nthreads+1;
      reflectVelocity<<<Nblocks, Nthreads>>>(DPD_group->getIndexIterator(access::location::gpu),
		      pos.raw(),
		      vel.raw(),
		      max_z,
		      N_dpd);
    }
    verlet->forwardTime();

    //Write results
    if(printSteps>0 and j%printSteps==0)
    {
      sys->log<System::DEBUG1>("[System] Writing to disk...");
      //continue;
      auto pos = pd->getPos(access::location::cpu, access::mode::read);
      auto vel = pd->getVel(access::location::cpu, access::mode::read);
      const int * sortedIndex = pd->getIdOrderedIndices(access::location::cpu);
      posOut<<"#Lx="<<0.5*box.boxSize.x<<";Ly="<<0.5*box.boxSize.y<<";Lz="<<0.5*box.boxSize.z<<";"<<endl;
      real3 p;
      int nbins = nbinsHisto;
      std::vector<double> histogram(nbins,0);
      std::vector<int> counter(nbins, 0);
      fori(0,N){
	real4 pc = pos.raw()[sortedIndex[i]];
	real3 v = vel.raw()[sortedIndex[i]];
	p = make_real3(pc); //box.apply_pbc(make_real3(pc));
	auto p_pbc = box.apply_pbc(make_real3(pc));
	int type = pc.w;
	float radius=0.5;
	if(type==0){ radius = 0.5; type=5;}
	
	posOut<<p<<" "<<radius<<" "<<type<<"\n";
	velOut<<v<<"\n";
	
	real z = p_pbc.z;

	int bin = int(((z+boxSize.z*0.5)/boxSize.z)*nbins+0.5);
	counter[bin]++;
	histogram[bin] += v.x;
      }
      histoOut<<""<<endl;
      fori(0, nbins){
	real time = j*dt;
	real z = (boxSize.z*(i/(double)nbins));
	real kvis = viscosityDPD/(densityDPD);
	real kappa = sqrt(2*M_PI*wallOscilationWaveNumber/(2*kvis));
	//std::cerr<<2*M_PI/kappa<<std::endl;
	real theo = 5*exp(-z*kappa)*cos(2*M_PI*wallOscilationWaveNumber*time- kappa*z);
	if(counter[i] > 0)
	  histoOut<<z<<" "<<histogram[i]/counter[i]<<" "<<theo<<"\n";
      }
      posOut<<flush;
      velOut<<flush;
      histoOut<<flush;
    }    
    //Sort the particles every few steps
    //It is not an expensive thing to do really.
    if(j%1000 == 0){
       pd->sortParticles();
    }
  }
  
  auto totalTime = tim.toc();
  sys->log<System::MESSAGE>("mean FPS: %.2f", numberSteps/totalTime);
  //sys->finish() will ensure a smooth termination of any UAMMD module.
  sys->finish();

  return 0;
}

//Read parameters from the file datamain
void readParameters(std::string datamain, shared_ptr<System> sys){
  InputFile in(datamain, sys);
  in.getOption("temperature",       InputFile::Required)>>temperature;
  in.getOption("gammaDPD",          InputFile::Required)>>gammaDPD;
  in.getOption("intensityDPD",      InputFile::Required)>>intensityDPD;
  in.getOption("numberParticlesDPD",InputFile::Required)>>numberParticlesDPD;
  in.getOption("boxSize",           InputFile::Required)>>boxSize.x>>boxSize.y>>boxSize.z;
  in.getOption("topWall_z",         InputFile::Required)>>topWall_z;
  in.getOption("cutOffWall_fluid",  InputFile::Required)>>cutOffWall_fluid;
  in.getOption("cutOffDPD",         InputFile::Required)>>cutOffDPD;
  in.getOption("wallBondsFile",     InputFile::Required)>>wallBondsFile;
  in.getOption("wallBondsFileFP",   InputFile::Required)>>wallBondsFileFP;
  in.getOption("wallParticleMass",  InputFile::Required)>>wallParticleMass;
  in.getOption("relaxSteps",        InputFile::Required)>>relaxSteps;
  in.getOption("numberSteps",       InputFile::Required)>>numberSteps;
  in.getOption("printSteps",        InputFile::Required)>>printSteps;
  in.getOption("dt",                InputFile::Required)>>dt;
  in.getOption("outputName",        InputFile::Required)>>outputName;
  in.getOption("wallCoordinatesFile",InputFile::Required)>>wallCoordinatesFile;
  in.getOption("wallOscilationAmplitude",InputFile::Required)>>wallOscilationAmplitude;
  in.getOption("wallOscilationWaveNumber",InputFile::Required)>>wallOscilationWaveNumber;
  in.getOption("epsilonWall",InputFile::Required)>>epsilonWall;
  in.getOption("nbinsHisto", InputFile::Optional)>>nbinsHisto;
}
