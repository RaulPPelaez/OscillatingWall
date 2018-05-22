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
#include"Integrator/VerletNVE.cuh"
#include"Interactor/NeighbourList/CellList.cuh"
#include"Interactor/PairForces.cuh"
#include"Interactor/ExternalForces.cuh"
#include"Interactor/BondedForces.cuh"
#include"Interactor/AngularBondedForces.cuh"
#include"Interactor/Potential/Potential.cuh"
#include"Interactor/Potential/DPD.cuh"
#include"utils/InitialConditions.cuh"
#include"utils/InputFile.h"
#include<fstream>

double epsilonWall = 5.0;

using namespace uammd;
//This external force takes care of making the wall oscilate (Also applies a constant force in the x direction if w=0)
struct WallOscilation: public ParameterUpdatable{
  real w;
  real A;
  real t;
  real relaxTime = -1;
  WallOscilation(real w, real A):w(w),A(A){t=0;}
  __host__ __device__ real3 force(const real4 &pos){
    
    real3 f = make_real3(A*(w==real(0.0)?real(1.0):sinf(w*t)),0, 0);
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

//This interactor computes the LJ interaction between the bottom wall particles and the DPD particles.
class WallFluidForces: public Interactor{

public:
  struct Parameters{
    Box box;
    real cutOff;
    real wallWidth;
    real wallFluidInterface;   
  };
  using PairForces = PairForces<Potential::LJ>;
private:
  shared_ptr<ParticleGroup> fluid_group, wall_group, particlesNearWall_group;
  thrust::device_vector<int> particlesToAdd;
  Parameters par;
  shared_ptr<PairForces> pf;
  
  void *cub_tmp_storage=nullptr;
  size_t cub_tmp_size = 0;
public:
  WallFluidForces(shared_ptr<ParticleData> pd,
		  shared_ptr<ParticleGroup> fluid_group,
		  shared_ptr<ParticleGroup> wall_group,
		  shared_ptr<System> sys,
		  WallFluidForces::Parameters par):
    Interactor(pd, sys, "WallFluidForces"),
    fluid_group(fluid_group), wall_group(wall_group), par(par){
    sys->log<System::MESSAGE>("[WallFluidForces] Initialized");
    sys->log<System::MESSAGE>("[WallFluidForces] %d particles in the wall", wall_group->getNumberParticles());
    sys->log<System::MESSAGE>("[WallFluidForces] %d particles in fluid", fluid_group->getNumberParticles());

    particlesNearWall_group = std::make_shared<ParticleGroup>(particle_selector::None(),
							      pd, sys, "Particles near wall");

    real3 L = par.box.boxSize;
    L.z = par.wallWidth+3*par.cutOff;
    
    PairForces::Parameters params;
    params.box = Box(L);
    auto pot = std::make_shared<Potential::LJ>(sys);
    {
      //Each Potential describes the pair interactions with certain parameters.
      //The needed ones are in InputPairParameters inside each potential, in this case:
      Potential::LJ::InputPairParameters par;
      par.epsilon = epsilonWall;
      par.shift = false;

      par.sigma = 1;
      par.cutOff = 2.5*par.sigma;
      //Once the InputPairParameters has been filled accordingly for a given pair of types,
      //a potential can be informed like this:

      pot->setPotParameters(0, 1, par);
      pot->setPotParameters(1, 1, par);
      par.epsilon = 0.0;
      pot->setPotParameters(0, 0, par);
    }

    pf = std::make_shared<PairForces>(pd, particlesNearWall_group, sys, params, pot);    
        
  }

  struct NearSelector{
    real max_z;
    real4* pos;
    NearSelector(real4 * pos, real max_z): pos(pos), max_z(max_z){}

    __device__ inline bool operator()(const int &i) const{ return pos[i].z<max_z;}
      

  };

  void sumForce(cudaStream_t st){
    particlesNearWall_group->clear();
    particlesNearWall_group->addParticlesByCurrentIndex(access::location::gpu, wall_group->getIndicesRawPtr(access::location::gpu), wall_group->getNumberParticles());

    particlesToAdd.resize(fluid_group->getNumberParticles());

    int *particlesToAdd_ptr = thrust::raw_pointer_cast(particlesToAdd.data());

    static int *numberSelectedParticles = nullptr;
    if(!numberSelectedParticles){
      cudaMallocManaged(&numberSelectedParticles, sizeof(int));
    }
    
    real max_z = -par.box.boxSize.z/2+par.wallWidth+2*par.cutOff;
    auto posGlobal = pd->getPos(access::location::gpu, access::mode::read);
    const int * indexFluid = fluid_group->getIndicesRawPtr(access::location::gpu);
    
    if(!cub_tmp_storage){
      sys->log<System::DEBUG1>("[WallFluidForces] Allocating cub storage");
      cub::DeviceSelect::If(cub_tmp_storage, cub_tmp_size,
			    indexFluid, particlesToAdd_ptr,
			    numberSelectedParticles, fluid_group->getNumberParticles(), NearSelector(posGlobal.raw(), max_z));
      cudaMalloc(&cub_tmp_storage, cub_tmp_size);
    }
    sys->log<System::DEBUG1>("[WallFluidForces] Looking for near fluid particles");
    cub::DeviceSelect::If(cub_tmp_storage, cub_tmp_size,
			  indexFluid, particlesToAdd_ptr,
			  numberSelectedParticles, fluid_group->getNumberParticles(), NearSelector(posGlobal.raw(), max_z));    
    cudaDeviceSynchronize();
    int nselect = *numberSelectedParticles;
    //cudaMemcpy(&nselect, numberSelectedParticles, sizeof(int), cudaMemcpyDeviceToHost);
    sys->log<System::DEBUG1>("[WallFluidForces] Adding %d particles to group", nselect);
    
    particlesNearWall_group->addParticlesByCurrentIndex(access::location::gpu,
							particlesToAdd_ptr,
							nselect);    
    pf->sumForce(0);
  }
  real sumEnergy(){return 0;}


};


//DPD dissipation between wall and fluid particles
struct WallDissipation{
  real gammaDPD, gammaWall_Fluid;
  inline __device__ real dissipativeStrength(int i, int j, const real4 &pi, const real4 &pj, ...) const{
    bool sameType = int(pi.w)==int(pj.w);
    bool wallwall = int(pi.w)==1;
    return (gammaDPD*sameType + gammaWall_Fluid*(!sameType))*(!wallwall);
  }
};

//Same as Harmonic, but the z position is shifted. This is used for the wall fixed bonds, allowing to displace the wall in the z direction.
struct HarmonicZ: public BondedType::Harmonic{
  real offset;
  HarmonicZ(real offset): offset(offset){}
  inline __device__ real3 force(int i, int j, const real3 &r12, const BondInfo &bi){    
    real3 r12z = make_real3(r12.x, r12.y, r12.z-offset);
    return Harmonic::force(i, j, r12z, bi);
  }    
};
//Confines only z direction
struct HarmonicZPure: public BondedType::Harmonic{
  real offset;
  HarmonicZPure(real offset): offset(offset){}
  inline __device__ real3 force(int i, int j, const real3 &r12, const BondInfo &bi){    
    real3 r12z = make_real3(0, 0, r12.z-offset);
    return Harmonic::force(i, j, r12z, bi);
  }    
};

using namespace std;

template<class GroupIndexIterator>
__global__ void imposeWallVelocityKernel(GroupIndexIterator groupIndex, real3 *vel, real w, real A, real t, int N){
  int id = blockIdx.x*blockDim.x + threadIdx.x;
  if(id>=N) return;
  int index = groupIndex[id];
  vel[index].x = A*sinf(w*t);
}
void imposeWallVelocity(shared_ptr<ParticleGroup> wall_group, shared_ptr<ParticleData> pd, real w, real A, real t){
      auto vel = pd->getVel(access::location::gpu, access::mode::readwrite);
      int N_wall = wall_group->getNumberParticles();
      int Nthreads = 64;
      int Nblocks  = N_wall/Nthreads+1;
      imposeWallVelocityKernel<<<Nblocks, Nthreads>>>(wall_group->getIndexIterator(access::location::gpu),
						      vel.raw(),
						      2*M_PI*w,
						      A,
						      t,
						      N_wall);
}


//Parameters
int numberParticlesDPD;
real cutOffWall_fluid;  //cut off between particles in the wall and the fluid (also wall-wall)
real cutOffDPD;         //cut off between fluid particles
real temperature;
real gammaDPD, gammaWall_Fluid;
real intensityDPD;

real3 boxSize;
real topWall_z;         //From 0 to Lz, Particles wont be allowed above this height

real wallParticleMass;

real wallOscilationAmplitude;
real wallOscilationWaveNumber;

int relaxSteps, numberSteps, printSteps;
real dt;

int nbinsHisto = 100;

std::string wallCoordinatesFile, wallBondsFile, wallBondsFileFP, wallAngularBondsFile;

std::string outputName;

bool printWallForces = false, printVelocities = false, printFluid = false;

//Ring down experiment, the wall starts with a force in the x direction, and it is released after the relaxation
bool ringDown = false;
//impose a velocity in the wall instead of a force
bool imposeVelocity = false;

void readParameters(std::string datamain, shared_ptr<System> sys);

int main(int argc, char *argv[]){
  auto sys = make_shared<System>(argc, argv);
  readParameters("data.main.dpd", sys);

  int numberParticlesToRead;
  int numberParticlesWall;

  ifstream inWall(wallCoordinatesFile);
  
  inWall>>numberParticlesToRead;

  // int numberParticlesObjects = numberStrands*particlesPerStrand;  
  // int N = numberParticlesDPD + numberParticlesWall + numberParticlesObjects;

  int N = numberParticlesDPD + numberParticlesToRead;
  ullint seed = 0xf31337Bada55D00dULL^time(NULL);
  sys->rng().setSeed(seed);

  auto pd = make_shared<ParticleData>(N, sys);

  real wallWidth;
  Box box(boxSize);//({40, 40, 90});
  real densityDPD = numberParticlesDPD/box.getVolume();

  real max_z = topWall_z-boxSize.z/2;
  //Initial configuration
  {
    auto pos = pd->getPos(access::location::cpu, access::mode::write);
    auto mass = pd->getMass(access::location::cpu, access::mode::write);

    real wallmaxz = -100000;
    real wallminz = 10000;
    real4 * p = pos.raw();
    fori(0,numberParticlesToRead){
      inWall>>p[i].x>>p[i].y>>p[i].z>>p[i].w;
      int type = int(p[i].w);
      if(type == 1){
	mass.raw()[i] = wallParticleMass;
	wallmaxz = std::max(p[i].z, wallmaxz);
	wallminz = std::min(p[i].z, wallminz);
      }
      else mass.raw()[i] = 1;
    }
    wallWidth = wallmaxz - wallminz;
    real min_z = wallmaxz+pow(2,1/6.);//cutOffWall_fluid;
    bool checkOverlaps = false;
    fori(numberParticlesToRead, N){
      p[i] = make_real4(sys->rng().uniform3(-box.boxSize.x*0.5,
					    box.boxSize.x*0.5),
			0);
      p[i].z = sys->rng().uniform(min_z, max_z);
      if(checkOverlaps){
	forj(0, numberParticlesToRead){
	  real3 rij = box.apply_pbc(make_real3(p[i]-p[j]));
	  if(dot(rij, rij) < cutOffWall_fluid*cutOffWall_fluid){
	    i--;
	    break;
	  }
	}
      }
      mass.raw()[i] = 1;
    }
    auto vel = pd->getVel(access::location::cpu, access::mode::write);
    double3 vcm = make_double3(0);
    fori(0,N){
      vel.raw()[i] = make_real3(sys->rng().gaussian(0, 0.01),
				sys->rng().gaussian(0, 0.01),
				sys->rng().gaussian(0, 0.01));
      vcm += make_double3(vel.raw()[i]);
    }
    vcm /= N;
    fori(0,N) vel.raw()[i] -= make_real3(vcm);    
  }
  
  auto all_group = make_shared<ParticleGroup>(pd, sys, "All");
  auto Wall_group = make_shared<ParticleGroup>(particle_selector::Type(1), pd, sys, "Wall_particles");
  numberParticlesWall = Wall_group->getNumberParticles();
  auto DPD_group = make_shared<ParticleGroup>(particle_selector::Type(0), pd, sys, "DPD_particles");
  auto Objects_group = make_shared<ParticleGroup>(particle_selector::Type(2), pd, sys, "Object_particles");
  int numberParticlesObjects = Objects_group->getNumberParticles();
  
  using NVE = VerletNVE;
  
  NVE::Parameters par;
  par.dt = dt;
  par.initVelocities = false;

  auto verlet = make_shared<NVE>(pd, all_group, sys, par);
  real viscosityDPD;
  {//DPD
    using DPD = Potential::DPD_impl<WallDissipation>;
    using PairForces = PairForces<DPD>;
  
    //This is the general interface for setting up a potential  
    DPD::Parameters dpd_params;
    dpd_params.cutOff = cutOffDPD;
    dpd_params.temperature = temperature;
    WallDissipation gamma;
    gamma.gammaDPD = gammaDPD;
    gamma.gammaWall_Fluid = gammaWall_Fluid;
    dpd_params.gamma = gamma;

    
    if(intensityDPD < 0) //This gives compresibility of water  1/k ~ 16
      dpd_params.A = 75.0*temperature/(pow(cutOffDPD, 4)*densityDPD);
    else
      dpd_params.A = intensityDPD;
    dpd_params.dt = par.dt;
    
    sys->log<System::MESSAGE>("DPD density: %f", densityDPD);
    
    viscosityDPD = 45.0/(4.0*M_PI)*temperature/(gammaDPD*pow(cutOffDPD,3)) +
      (2.0*M_PI)/(1575.0)*pow(densityDPD,2)*gammaDPD*pow(cutOffDPD,5);
    sys->log<System::MESSAGE>("DPD viscosity: %f", viscosityDPD);
    
    auto pot = make_shared<DPD>(sys, dpd_params);

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
  
  if(!imposeVelocity){//Wall Fixed point bonds
    using BondedForces = BondedForces<HarmonicZ>;
    //You can use Elastic_Network_Model.cpp to generate some example bonds for the starting configuration.
    BondedForces::Parameters params;
    params.file = wallBondsFileFP.c_str();  //Box to work on
    auto bondedforces = make_shared<BondedForces>(pd, sys, params,
						  HarmonicZ(0));
    verlet->addInteractor(bondedforces);
  }
  else{
    using BondedForces = BondedForces<HarmonicZPure>;
    //You can use Elastic_Network_Model.cpp to generate some example bonds for the starting configuration.
    BondedForces::Parameters params;
    params.file = wallBondsFileFP.c_str();  //Box to work on
    auto bondedforces = make_shared<BondedForces>(pd, sys, params,
						  HarmonicZPure(0));
    verlet->addInteractor(bondedforces);

  }

  {//Object Angular bonds
    //You can use Elastic_Network_Model.cpp to generate some example bonds for the starting configuration.
    AngularBondedForces::Parameters params;
    params.readFile = wallAngularBondsFile.c_str();  //Box to work on
    params.box = box;
    auto bondedforces = make_shared<AngularBondedForces>(pd, sys, params);
    verlet->addInteractor(bondedforces);
  }
  {
    using PairForces = PairForces<Potential::LJ>;
    PairForces::Parameters params;
    params.box = box;
    auto pot = std::make_shared<Potential::LJ>(sys);
    {
      //Each Potential describes the pair interactions with certain parameters.
      //The needed ones are in InputPairParameters inside each potential, in this case:
      Potential::LJ::InputPairParameters par;
      par.epsilon = epsilonWall;
      par.shift = false;

      par.sigma = 1;
      par.cutOff = 2.5*par.sigma;
      //Once the InputPairParameters has been filled accordingly for a given pair of types,
      //a potential can be informed like this:
      pot->setPotParameters(2, 2, par);
    }
    auto pf = std::make_shared<PairForces>(pd, Objects_group, sys, params, pot);    
    verlet->addInteractor(pf);

  }
  {//Wall-Wall and Wall-fluid
    WallFluidForces::Parameters par;
    par.box = box;
    par.cutOff = cutOffWall_fluid;
    par.wallWidth = wallWidth;
    par.wallFluidInterface = -box.boxSize.z/2;
    auto wallFluidForces  = make_shared<WallFluidForces>(pd, DPD_group, Wall_group, sys, par);    
    verlet->addInteractor(wallFluidForces);
  }


  real wallDensity = numberParticlesWall*wallParticleMass/(box.boxSize.y*box.boxSize.x);
  sys->log<System::MESSAGE>("C: %f", viscosityDPD*densityDPD/(2*M_PI*wallOscilationWaveNumber*pow(wallDensity,2)));
  sys->log<System::MESSAGE>("Penetration length: %f", sqrt(2*viscosityDPD/(2*M_PI*wallOscilationWaveNumber*densityDPD)));
							      

  sys->log<System::MESSAGE>("RUNNING!!!");

  pd->sortParticles();

  ofstream posOut(outputName+".pos");
  ofstream velOut;  if(printVelocities) velOut.open(outputName+".vel");
  ofstream forceOut; if(printWallForces) forceOut.open(outputName+".wallForces");
  ofstream histoOut(outputName+".histo");

  Timer tim;
  tim.tic();
  //Wall Oscilation
  real OscilationAmplitude = 0;
  real w = 0;
  if(ringDown){
    OscilationAmplitude = wallOscilationAmplitude;
  }
  auto wallOscilationForce = make_shared<WallOscilation>(w,OscilationAmplitude);
  {
    auto wallOscilation = make_shared<ExternalForces<WallOscilation>>(pd, Wall_group, sys,
								      wallOscilationForce);
    if(!imposeVelocity)
      verlet->addInteractor(wallOscilation);
  }
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
  sys->log<System::MESSAGE>("RUNNING!!!");
  cudaDeviceSynchronize();
  //If this is a force oscillation simulation activate oscilation
  if(!ringDown && ! imposeVelocity){
    w = 2*M_PI*wallOscilationWaveNumber;
    wallOscilationForce->w = w;
    wallOscilationForce->A = wallOscilationAmplitude;
  }
  //if this is a ringDown sim. deactivate force.
  else if(ringDown){
    wallOscilationForce->A = 0;
  }
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
    if(imposeVelocity){
      imposeWallVelocity(Wall_group, pd, wallOscilationWaveNumber, wallOscilationAmplitude, j*dt);
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
	if(type==0 && !printFluid) continue;
	float radius=0.5;
	if(type==0){ radius = 0.2; type=5;}
	if(type==2){ type = 6; }	
	posOut<<p<<" "<<radius<<" "<<type<<"\n";
	if(printVelocities) velOut<<v<<"\n";

	real z = p_pbc.z;

	int bin = int(((z+boxSize.z*0.5)/boxSize.z)*nbins+0.5);
	counter[bin]++;
	histogram[bin] += v.x;
      }
      if(printWallForces){
	auto force = pd->getForce(access::location::cpu, access::mode::read);
	auto wallParticlesIndices = Wall_group->getIndexIterator(access::location::cpu);
	real3 totalForce = make_real3(0);
	fori(0, Wall_group->getNumberParticles()){
	  totalForce += make_real3(force.raw()[wallParticlesIndices[i]]);// + wallOscilationForce->force(pos.raw()[wallParticlesIndices[i]]);
	}

	forceOut<<totalForce<<endl;
	
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
      
      if(printVelocities) velOut<<flush;
      if(printWallForces) forceOut<<flush;
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
  in.getOption("gammaWall_Fluid",   InputFile::Required)>>gammaWall_Fluid;
  in.getOption("intensityDPD",      InputFile::Required)>>intensityDPD;
  in.getOption("numberParticlesDPD",InputFile::Required)>>numberParticlesDPD;
  in.getOption("boxSize",           InputFile::Required)>>boxSize.x>>boxSize.y>>boxSize.z;
  in.getOption("topWall_z",         InputFile::Required)>>topWall_z;
  in.getOption("cutOffWall_fluid",  InputFile::Required)>>cutOffWall_fluid;
  in.getOption("cutOffDPD",         InputFile::Required)>>cutOffDPD;
  in.getOption("wallBondsFile",     InputFile::Required)>>wallBondsFile;
  in.getOption("wallBondsFileFP",   InputFile::Required)>>wallBondsFileFP;
  in.getOption("wallAngularBondsFile",InputFile::Required)>>wallAngularBondsFile;
  in.getOption("wallParticleMass",  InputFile::Required)>>wallParticleMass;

  in.getOption("relaxSteps",        InputFile::Required)>>relaxSteps;
  in.getOption("numberSteps",       InputFile::Required)>>numberSteps;
  in.getOption("printSteps",        InputFile::Required)>>printSteps;
  in.getOption("dt",                InputFile::Required)>>dt;
  in.getOption("outputName",        InputFile::Required)>>outputName;
  in.getOption("wallCoordinatesFile",InputFile::Required)>>wallCoordinatesFile;
  in.getOption("wallOscilationAmplitude",InputFile::Required)>>wallOscilationAmplitude;
  in.getOption("wallOscilationWaveNumber",InputFile::Required)>>wallOscilationWaveNumber;
  in.getOption("nbinsHisto", InputFile::Optional)>>nbinsHisto;
  
  
  printVelocities = bool(in.getOption("printVelocities",InputFile::Optional));
  printWallForces = bool(in.getOption("printWallForces",InputFile::Optional));
  printFluid = bool(in.getOption("printFluid",InputFile::Optional));
  ringDown = bool(in.getOption("ringDown",InputFile::Optional));
  imposeVelocity = bool(in.getOption("imposeVelocity",InputFile::Optional));

  if(ringDown && imposeVelocity){
    sys->log<System::CRITICAL>("You cannot impose velocity in a ringdown simulation!");
  }
  if(ringDown)
    sys->log<System::MESSAGE>("ringDown mode!");
  if(imposeVelocity)
    sys->log<System::MESSAGE>("imposeVelocity mode!");
    
}
