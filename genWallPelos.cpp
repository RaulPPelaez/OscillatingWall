
#include<vector>
#include<random>
#include<fstream>
#include<functional>
#include<cstdio>
#include<cstring>
#include<iostream>






std::vector<double> genWall(double Lx, double Ly, int nlayers, double a, double lowerZ){

  std::vector<double> layer;  

  int nx = int(Lx/a+0.5);
  int ny = int((Ly/a)/(sqrt(3)*0.5)+0.5);
  if(ny%2 != 0) ny-=1;
  double dr = Lx/nx*0.5;
  double dry = Ly/ny/sqrt(3);
  int p = 0;
  layer.resize(3*nx*ny,0);
  for(int i = 0; i<nx; i++){
    for(int j = 0; j<ny; j++){
      layer[3*p+0] = -Lx*0.5+ i*dr*2+((1-pow(-1.0, j))*0.5)*dr;
      layer[3*p+1] = -Ly*0.5+ sqrt(3)*j*dry;
      p++;
    }
  }

  auto wall = layer;
  wall.resize(layer.size()*nlayers);
  
  for(int k = 0; k<nlayers; k++){
    float sign = pow(-1, k);
    if(k==0) sign=0;
    for(int i = 0; i<layer.size()/3; i++){
      layer[3*i+2] += dr*sqrt(6)/3.*2;
      layer[3*i+1] += dry*sign*sqrt(3)/3.;
      layer[3*i+0] += dr*sign;
      
      wall[layer.size()*k + 3*i + 0] = layer[3*i + 0];
      wall[layer.size()*k + 3*i + 1] = layer[3*i + 1];
      wall[layer.size()*k + 3*i + 2] = layer[3*i + 2];
    }
  }

    int N = wall.size()/3;
    double cm[3] = {0,0,0};
    double minZ = 100000;
    for(int i = 0; i<N; i++){
      for(int j = 0; j<3; j++) cm[j] += wall[3*i+j];
      minZ = std::min(wall[3*i+2], minZ);
    }
    cm[0] /= N;
    cm[1] /= N;
    cm[2] /= N;
    minZ -= cm[2];
    for(int i = 0; i<N; i++){
      for(int j = 0; j<3; j++) wall[3*i+j] -= cm[j];
      wall[3*i + 2] -= minZ;
      wall[3*i + 2] += lowerZ;
    }
    return wall;
}


std::vector<double> genPelo(int NperPelo, double a, double Zbase){

  std::vector<double> pelo(3*3*NperPelo,0);
  for(int i = 0; i<NperPelo; i++){
    pelo[3*i + 2] = a*i;

    pelo[3*(i+NperPelo) + 0] = sqrt(3)*0.5*a;
    pelo[3*(i+NperPelo) + 2] = a*i+0.5*a;

    pelo[3*(i+2*NperPelo) + 0] = a*sqrt(3.0)/6.0;
    pelo[3*(i+2*NperPelo) + 1] = a*sqrt(2/3.);
    pelo[3*(i+2*NperPelo) + 2] = a*i+0.5*a;
  }

  for(int i = 0; i<pelo.size()/3; i++){
    pelo[3*i+2] += Zbase;
  }
  return pelo;
}


  


int main(int argc, char *argv[]){


  std::mt19937 g1(time(NULL));
  std::uniform_real_distribution<double> uniform_dist(-0.5,0.5);
  auto uniform = std::bind(uniform_dist, g1);
  
  
  double Lx = std::stod(argv[1]);
  double Ly = std::stod(argv[2]);
  double Lz = std::stod(argv[3]);
  int nlayers = std::atoi(argv[4]);  
  double a = std::stod(argv[5]);

  int npelos = std::atoi(argv[6]);
  int NperPelo = std::atoi(argv[7]);
  
  double minZ = -Lz*0.5+1;
  auto wall = genWall(Lx, Ly, nlayers, a, minZ);
  double Zbase = -Lz*0.5+2;
  auto pelo = genPelo(NperPelo, a, Zbase);

  std::ofstream pout("init.pos");
  std::ofstream b3out("bonds3.dat");
  pout<<npelos*NperPelo*3+wall.size()/3<<std::endl;
  b3out<<npelos*3*(NperPelo-2)<<std::endl;
  

  std::vector<double> peloPos;
  for(int i = 0; i<npelos; i++){
    double xpos = uniform()*Lx;
    double ypos = uniform()*Ly;

    bool success = false;
    if(peloPos.size()>0)
    while(!success){
      xpos = uniform()*Lx;
      ypos = uniform()*Ly;	
      for(int j = 0; j<peloPos.size()/2; j++){    
	double r = sqrt(pow(peloPos[2*j]-xpos, 2) + pow(peloPos[2*j+1] - ypos, 2));
	if(r<4*a){success = false; break;}
	success = true;
      }
    }
    peloPos.push_back(xpos);
    peloPos.push_back(ypos);
    for(int j = 0; j<NperPelo*3; j++){
      pout<<pelo[3*j]+xpos<<" "<<pelo[3*j + 1] + ypos<<" "<<pelo[3*j + 2]<<" 2\n";
    }
    
    int baseIndex = i*NperPelo*3;
    for(int k=0; k<3; k++){
      for(int j = 0; j<NperPelo-2; j++){
	b3out<<NperPelo*k+baseIndex+j<<" "<<NperPelo*k+baseIndex+j+1<<" "<<NperPelo*k+baseIndex+j+2<<"\n";
      }
    }
    
  }

  for(int i = 0; i<wall.size()/3; i++){
    for(int j = 0; j<3; j++) pout<<wall[3*i+j]<<" ";
    pout<<"1\n";

  }

  

  

  


  return 0;
}
