





datamain=data.main.dpd

boxSize=$(grep "^boxSize" $datamain | awk '{print $2, $3, $4}')


Lx=$(echo $boxSize | awk '{print $1}')
Ly=$(echo $boxSize | awk '{print $2}')
Lz=$(echo $boxSize | awk '{print $3}')
nlayers=3
radius=$(echo 1 | awk '{print 2^(1/6.)}')


npelos=3
NperPelo=20
angK=10000
g++ -std=c++11 -O3  genWallPelos.cpp -o genWallPelos



./genWallPelos $Lx $Ly $Lz $nlayers $radius $npelos $NperPelo

awk 'NF==3{print $0, '$angK',0}NF==1' bonds3.dat > init.3bonds




g++ -std=c++11 -O3 Elastic_Network_Model.cpp -o ENM
./ENM init.pos 1.2 300 $Lx $Ly 10000 > init.2bonds
Nwall=$(awk '$4==1' init.pos | wc -l)

echo 0 > init.bondsFP
echo $Nwall >> init.bondsFP

awk '$4==1{print NR-2, $0}' init.pos | awk '{print $1, $2,$3,$4, 200, 0}' >> init.bondsFP


rm -f genWallPelos ENM
