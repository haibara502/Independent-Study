#!bin/bash

for file in ./stories/*
do
	if test -d $file
	then
#echo $file
		sizes="$(wc -c <$file/*script*)"
		if [ "$sizes" ==  0 ]
		then
			echo $file
		fi
	fi
done

