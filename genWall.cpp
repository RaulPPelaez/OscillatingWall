


#include"uammd/src/third_party/bravais/bravais.h"

#include<cstdio>
#include<iostream>
#include<vector>
#include<limits.h>
using namespace std;

template<class T>
T norm( T* v){
  return v[0]*v[0]+v[1]*v[1]+v[2]*v[2];
}
int main(int argc, char* argv[]){

  double Lx = std::stod(argv[1]);
  double Ly = std::stod(argv[2]);
  int nlayers = std::atoi(argv[3]);  
  double a = std::stod(argv[4]);
  
  int Ngrid = Lx*Ly/a/a;

  std::vector<float> layer(4*Ngrid, 0);
  
  Bravais(layer.data(), tri, Ngrid,
	  Lx/a, Ly/a, 0,
	  1.0, NULL, NULL, true);


  //auto pos = layer;

  vector<float> pos;

  
  pos.clear();
  int nx = int(Lx/a+0.5);
  int ny = int((Ly/a)/(sqrt(3)*0.5)+0.5);
  if(ny%2 != 0) ny-=1;
  double dr = Lx/nx*0.5;
  double dry = Ly/ny/sqrt(3);
  int p = 0;
  pos.resize(4*nx*ny,0);
  for(int i = 0; i<nx; i++){

    for(int j = 0; j<ny; j++){
      pos[4*p+0] = -Lx*0.5+ i*dr*2+((1-pow(-1.0, j))*0.5)*dr;
      pos[4*p+1] = -Ly*0.5+ sqrt(3)*j*dry;
      pos[4*p+2] = 0;
      p++;
    }
  }
  
  std::cout<<nlayers*(pos.size()/4)<<"\n";
  
  //std::cout<<"#Lx="<<Lx*0.5<<";Ly="<<Ly*0.5<<";Lz="<<nlayers*0.5<<";"<<std::endl;

  //for(int i = 0; i<pos.size()/4; i++){        pos[4*i+2] += -nlayers*a*0.5; }
  for(int k = 0; k<nlayers; k++){
    float sign = pow(-1, k);
    if(k==0) sign=0;
    for(int i = 0; i<pos.size()/4; i++){
      pos[4*i+2] += dr*sqrt(6)/3.*2;
      pos[4*i+1] += dry*sign*sqrt(3)/3.;
      pos[4*i+0] += dr*sign;
      
      for(int j = 0; j<3;j++){
	std::cout<<pos[4*i+j]<<" ";
      }
      
      std::cout<<" 1\n";
    }
  }  
  return 0;
}
