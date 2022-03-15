
# Author: Rachel Bawden
# Contact: rachel.bawden@limsi.fr
# Last modif: 13/06/2017

python2=/path/to/py2.7/bin/python
mypython3=/path/to/py3.5/bin/python

SRC="fr"
TRG="en"

data_dir=/path/to/data/
opensubs_dir=$data_dir/OpenSubtitles2016
working_dir=/path/to/processed/data
SCRIPTS=/path/to/scripts-imsdb
champollion=/path/to/tools/champollion-1.2/


CTK=$champollion 
LC_ALL=C


# Make directories if necessary
for folder in opensubs_all opensubs_minusimsdb opensubs_train \
			  opensubs_dev imsdb imsdb/subtitles imsdb/speech imsdb/scripts \
			  imsdb/structured_scripts imsdb/alignments; do
	[ -d $working_dir/$folder ] || mkdir $working_dir/$folder
done


# get imdb numbers of available imsdb films
$python2 $SCRIPTS/crawl_imsdb.py \
    -n $working_dir/imsdb/all.nums-titles.json \
    $working_dir/imsdb/scripts/

# Remove imsdb films from Opensubs train set
# Get the line numbers of IMSDB films
echo ">> Getting line numbers of imsdb films in OpenSubs"
$python2 $SCRIPTS/get_imsdb_linenumbers.py \
	   $working_dir/imsdb/all.nums-titles.json \
 	   `wc -l $working_dir/opensubs_all/noblank.$SRC-$TRG.$SRC | perl -pe 's/^\s*//g' | cut -d" " -f 1` \
 	   -f  $working_dir/opensubs_all/opensubs.$SRC-$TRG.filminfo \
	   > $working_dir/imsdb/opensubs.imsdb.$SRC-$TRG.list

$python2 $SCRIPTS/get_imsdb_linenumbers.py \
	   $working_dir/imsdb/all.nums-titles.json \
 	   `wc -l $working_dir/opensubs_all/noblank.$SRC-$TRG.$SRC | perl -pe 's/^\s*//g' | cut -d" " -f 1` \
 	   -f  $working_dir/opensubs_all/opensubs.$SRC-$TRG.filminfo --printfilminfo \
 	   > $working_dir/imsdb/opensubs.imsdb.$SRC-$TRG.filminfo

# Get the line numbers of remaining OpenSubs films
echo ">> Getting line numbers of Opensubs minus imsdb films in OpenSubs"
$python2 $SCRIPTS/get_imsdb_linenumbers.py \
	   $working_dir/imsdb/all.nums-titles.json \
 	   `wc -l $working_dir/opensubs_all/noblank.$SRC-$TRG.$SRC | perl -pe 's/^\s*//g' | cut -d" " -f 1` \
 	   -f $working_dir/opensubs_all/opensubs.$SRC-$TRG.filminfo -v \
 	   > $working_dir/opensubs_minusimsdb/opensubs.minus-imsdb.$SRC-$TRG.list

$python2 $SCRIPTS/get_imsdb_linenumbers.py \
	   $working_dir/imsdb/all.nums-titles.json \
 	   `wc -l $working_dir/opensubs_all/noblank.$SRC-$TRG.$SRC | perl -pe 's/^\s*//g' | cut -d" " -f 1` \
 	   -f  $working_dir/opensubs_all/opensubs.$SRC-$TRG.filminfo --printfilminfo -v \
 	   > $working_dir/opensubs_minusimsdb/opensubs.minus-imsdb.$SRC-$TRG.filminfo

echo ">> Getting IMSDB films from OpenSubs2016 corpus (opensubs_minusimsdb/)"
# Get the sentences corresponding to opensubs\imsdb numbers
for lang in $SRC $TRG; do
	python $SCRIPTS/get-these-lines-from-numbers.py \
 		   $working_dir/opensubs_all/noblank.$SRC-$TRG.$lang \
 		   $working_dir/opensubs_minusimsdb/opensubs.minus-imsdb.$SRC-$TRG.list \
 		   > $working_dir/opensubs_minusimsdb/opensubs.minus-imsdb.$SRC-$TRG.$lang
done

echo ">> Getting just IMSDB from OpenSubs2016 corpus (imsdb/)"
# Get the sentences corresponding to opensubs\imsdb numbers
for lang in $SRC $TRG; do
	python $SCRIPTS/get-these-lines-from-numbers.py \
 		   $working_dir/opensubs_all/noblank.$SRC-$TRG.$lang \
 		   $working_dir/imsdb/opensubs.imsdb.$SRC-$TRG.list \
 		   > $working_dir/imsdb/opensubs.imsdb.$SRC-$TRG.$lang
done

# Remove 5000 films from OpenSubs\imsdb to use as dev if necessary
echo ">> Separating out last 5000 films from OpenSubs2016-IMSDB to make dev set (opensubs_train/ and opensubs_dev/)"


lastline=`tail -5000 $working_dir/opensubs_minusimsdb/opensubs.minus-imsdb.$SRC-$TRG.filminfo | head -1 | cut -f 2 `
firstfilms=`tail -5000 $working_dir/opensubs_minusimsdb/opensubs.minus-imsdb.$SRC-$TRG.filminfo | head -1 | cut -f 1 `

# get dev corpus
cat $working_dir/opensubs_minusimsdb/opensubs.minus-imsdb.$SRC-$TRG.$SRC \
 	| sed -n "${lastline},50000000000p" >  $working_dir/opensubs_dev/opensubs.dev.$SRC-$TRG.$SRC

cat $working_dir/opensubs_minusimsdb/opensubs.minus-imsdb.$SRC-$TRG.$TRG \
 	| sed -n "${lastline},50000000000p" >  $working_dir/opensubs_dev/opensubs.dev.$SRC-$TRG.$TRG

# get line numbers from opensubs_all
cat $working_dir/opensubs_minusimsdb/opensubs.minus-imsdb.$SRC-$TRG.list \
	| sed -n "${lastline},50000000000p" >  $working_dir/opensubs_dev/opensubs.dev.$SRC-$TRG.list

lastline=$(($lastline-1))
firstfilms=$(($firstfilms-1))

# get train corpus
cat $working_dir/opensubs_minusimsdb/opensubs.minus-imsdb.$SRC-$TRG.$SRC \
 	| sed -n "1,${lastline}p" >  $working_dir/opensubs_train/opensubs.train.$SRC-$TRG.$SRC

cat $working_dir/opensubs_minusimsdb/opensubs.minus-imsdb.$SRC-$TRG.$TRG \
 	| sed -n "1,${lastline}p" >  $working_dir/opensubs_train/opensubs.train.$SRC-$TRG.$TRG

cat $working_dir/opensubs_minusimsdb/opensubs.minus-imsdb.$SRC-$TRG.filminfo \
 	| sed -n "1,${firstfilms}p" >  $working_dir/opensubs_train/opensubs.train.$SRC-$TRG.filminfo

# get line numbers from opensubs_all
cat $working_dir/opensubs_minusimsdb/opensubs.minus-imsdb.$SRC-$TRG.list  \
	| sed -n "1,${firstfilms}p" >  $working_dir/opensubs_train/opensubs.train.$SRC-$TRG.list






		  
