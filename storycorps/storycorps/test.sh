#!bin/sh
name=""

for file in ./*
do
	if test -d $file
	then
		echo $file
		cd $file
		ls
		name=$(cat *mp3.txt)
		name=$name'3'
		wget $name
		cd ..
	fi
done
