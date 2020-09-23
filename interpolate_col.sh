set -e
# Folder in
di="/home/lunet/gytm3/Everest2019/Research/ReanalysisData/ERA5/"
# Folder out
do="/home/lunet/gytm3/Everest2019/Research/Weather/Data/"
for f in ${di}*.nc; do

	cmd="cdo -s remapbil,lon=86.9295_lat=27.9719 ${f} ${f/${di}/${do}}"
	${cmd}
	echo "Processed ${f}..."

done
