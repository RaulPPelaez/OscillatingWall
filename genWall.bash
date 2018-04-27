





Lx=40
Ly=40

nlayers=3
radius=$(echo 1 | awk '{print 2^(1/6.)}')

g++ -std=c++11 -O3 genWall.cpp -o genWall

./genWall $Lx $Ly $nlayers $radius > wall.init

ENM wall.init 1.2 200 $Lx $Ly 10000 > wall.bonds
N=$(head -1 wall.init)

echo 0 > wall.bondsFP
echo $N >> wall.bondsFP

tail -n+2 wall.init | awk '{print NR-1, $1,$2,$3, 200, 0}' >> wall.bondsFP


rm genWall
