library("raster")
library("horizon")
fi="/home/lunet/gytm3/Everest2019/Research/Weather/Data/dem_10.asc.txt"
res=5

dem<-raster(fi)
angi<-seq(1,360,5)
out<-vector()
out<-c(out,seq(1,length(angi)))*0

count<-1
for (i in angi){
  
  a=horizonSearch(dem, i, maxDist = 5000, degrees = TRUE,
                ll = FALSE, filename = "", blockSize = NULL)
  ango<-a[621,708]
  out[count]<-ango
  print(paste("Computed for angle:",i,"(horizon is:",ango,"degrees)"))
  count<-count+1
  
}