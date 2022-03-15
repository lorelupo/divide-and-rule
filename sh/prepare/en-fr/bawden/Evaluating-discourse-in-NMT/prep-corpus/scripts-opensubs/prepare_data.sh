
# Author: Rachel Bawden
# Contact: rachel.bawden@limsi.fr
# Last modif: 07/07/2017


#---------------------------------------------------------------------
# paths to your python distributions (python 3)
mypython3=/path/to/python3

# scripts and tools paths (path to scripts-opensubs/)
SCRIPTS=./

# source and target languages
SRC="fr"
TRG="en"

# where raw OpenSubtitles2016 data will be stored
data_dir=/path/to/data/
opensubs_dir=$data_dir/OpenSubtitles2016 

# where processed parallel data will be stored
working_dir=/path/to/processed/data

reload=false # change to true to redo all steps (otherwise will not repeat steps already done)

#---------------------------------------------------------------------
# Make directories if necessary
for folder in datadir $working_dir/opensubs_all; do
	[ -d $folder ] || mkdir $folder
done

# Download untokenised files (righthand side of matrix) for each language from Opus
if ([ ! -d $opensubs_dir/raw/$SRC ] && [ ! -f $opensubs_dir/$SRC.raw.tar.gz ]) || [ reload == "true" ]; then
	wget http://opus.lingfil.uu.se/download.php?f=OpenSubtitles2016/$SRC.raw.tar.gz \
		-O $data_dir/$SRC.raw.tar.gz
	tar -xzvf $data_dir/$SRC.raw.tar.gz
fi

if ([ ! -d $opensubs_dir/raw/$TRG ] && [ ! -f $opensubs_dir/$TRG.raw.tar.gz ]) || [ reload == "true" ]; then
	echo wget http://opus.lingfil.uu.se/download.php?f=OpenSubtitles2016/$TRG.raw.tar.gz \
		-O $data_dir/$TRG.raw.tar.gz
	echo tar -xzvf $data_dir/$TRG.raw.tar.gz 
fi

# Determine which language is alphabetically first (for name of alignment file)
align_src=`[ "$SRC" \< "$TRG" ] && echo "$SRC" || echo "$TRG"`
align_trg=`[ "$SRC" \< "$TRG" ] && echo "$TRG" || echo "$SRC"`

# Download alignment file for each language (ces)
if ([ ! -f $opensubs_dir/alignments.$align_src-$align_trg.xml.gz ] && \
	   [ ! -f $opensubs_dir/alignments.$SRC-$TRG.xml.gz ]) || [ reload == "true" ]; then
	wget http://opus.lingfil.uu.se/download.php?f=OpenSubtitles2016/xml/$align_src-$align_trg.xml.gz \
		-O $opensubs_dir/alignments.$SRC-$TRG.xml.gz	
fi

# Get the opensubs parallel corpus
echo ">> Preparing all opensubs2016 data (opensubs_all/)"
if ([ ! -f $working_dir/opensubs_all/raw.$SRC-$TRG.filminfo ] && \
		[ ! -f $working_dir/opensubs_all/opensubs.$SRC-$TRG.filminfo ]) || [ reload == "true" ]; then
	echo "Getting parallel corpus"
	$mypython3 $SCRIPTS/create-open-subs-corpus.py \
			   -r $opensubs_dir/raw \
			   -a $alignmentfile -o $working_dir/opensubs_all/raw.$SRC-$TRG \
			   -s $TRG -t $SRC \
			   > $working_dir/opensubs_all/raw.$SRC-$TRG.filminfo
fi

# Preclean (fix encodings and whitespace characters such as \r)
if  [ ! -f $working_dir/opensubs_all/precleaned.$SRC-$TRG.$SRC ] || [ reload == "true" ]; then
	echo "Precleaning $SRC"
	if  [[ "$SRC" == "en" ||  "$SRC" == "fr" ]]; then
		cat $working_dir/opensubs_all/raw.$SRC-$TRG.$SRC | \
			perl $SCRIPTS/fix_mixed_encodings.pl \
				 > $working_dir/opensubs_all/precleaned.$SRC-$TRG.$SRC
	else
		zcat $working_dir/opensubs_all/raw.$SRC-$TRG.$SRC | \
			perl -pe 's/\r//g' | \
			gzip > $working_dir/opensubs_all/precleaned.$SRC-$TRG.$SRC
	fi
fi
if  [ ! -f $working_dir/opensubs_all/precleaned.$SRC-$TRG.$TRG ] || [ reload == "true" ]; then
	echo "Precleaning $TRG"
	if  [[ "$TRG" == "en" ||  "$TRG" == "fr" ]]; then
		cat $working_dir/opensubs_all/raw.$SRC-$TRG.$TRG | \
			perl $SCRIPTS/fix_mixed_encodings.pl \
				 > $working_dir/opensubs_all/precleaned.$SRC-$TRG.$TRG
	else
		zcat $working_dir/opensubs_all/raw.$SRC-$TRG.$TRG | \
			perl -pe 's/\r//g' | \
			gzip > $working_dir/opensubs_all/precleaned.$SRC-$TRG.$TRG
	fi
fi

# Birecode to eliminate all extra problems
if  [ ! -f $working_dir/opensubs_all/birecoded.$SRC-$TRG.$SRC ] || [ reload == "true" ]; then
	echo "Birecoding $SRC"
	cat $working_dir/opensubs_all/precleaned.$SRC-$TRG.$SRC | recode -f u8..unicode \
		| recode unicode..u8 > $working_dir/opensubs_all/birecoded.$SRC-$TRG.$SRC
fi
if  [ ! -f $working_dir/opensubs_all/birecoded.$SRC-$TRG.$TRG ] || [ reload == "true" ]; then
	echo "Birecoding $TRG"
	cat $working_dir/opensubs_all/precleaned.$SRC-$TRG.$TRG | recode -f u8..unicode \
		| recode unicode..u8 > $working_dir/opensubs_all/birecoded.$SRC-$TRG.$TRG
fi
	
# Subtitle-specific cleaning (removed unwanted characters and sentences and correct some OCR problems)
if  [ ! -f $working_dir/opensubs_all/cleaned.$SRC-$TRG.$SRC ] || [ reload == "true" ]; then
	echo "Cleaning $SRC"
	$mypython3 $SCRIPTS/clean-up-subs.py \
			   $working_dir/opensubs_all/birecoded.$SRC-$TRG.$SRC $SRC \
			   > $working_dir/opensubs_all/cleaned.$SRC-$TRG.$SRC
fi
if  [ ! -f $working_dir/opensubs_all/cleaned.$SRC-$TRG.$TRG ] || [ reload == "true" ]; then
	echo "Cleaning $TRG"
	$mypython3 $SCRIPTS/clean-up-subs.py \
			   $working_dir/opensubs_all/birecoded.$SRC-$TRG.$TRG $TRG \
			   > $working_dir/opensubs_all/cleaned.$SRC-$TRG.$TRG
fi

# Remove blank lines and recalculate the film line info
if  ([ ! -f $working_dir/opensubs_all/noblank.$SRC-$TRG.$SRC ] && \
		[! -f $working_dir/opensubs_all/noblank.$SRC-$TRG.$TRG ]) || \
		[ ! -f $working_dir/opensubs_all/opensubs.$SRC-$TRG.filminfo ] || [ reload == "true" ]; then
	echo "Removing blank lines"
	$mypython3 $SCRIPTS/filter-empty-lines.py \
			   $working_dir/opensubs_all/cleaned.$SRC-$TRG.$SRC \
			   $working_dir/opensubs_all/cleaned.$SRC-$TRG.$TRG \
			   $working_dir/opensubs_all/noblank.$SRC-$TRG.$SRC \
			   $working_dir/opensubs_all/noblank.$SRC-$TRG.$TRG \
			   > $working_dir/tmpfilmlines
	echo "Recalculating film info"
	$mypython3 $SCRIPTS/recalculate-film-lines.py \
			   $working_dir/opensubs_all/raw.$SRC-$TRG.filminfo \
			   $working_dir/tmpfilmlines \
	 		   > $working_dir/opensubs_all/opensubs.$SRC-$TRG.filminfo
	rm $working_dir/tmpfilmlines
fi

TAB=$'\t' 
# Add line numbers to "noblank files" to keep a track of them later
if [ ! -f $working_dir/opensubs_all/noblank.numbered.$SRC-$TRG.$SRC ] || [ reload == "true" ]; then
	echo "Numbering noblank $SRC file"
	sed = $working_dir/opensubs_all/noblank.$SRC-$TRG.$SRC | sed -e "N;s/\n/${TAB}/" > $$;
	cat $$ > $working_dir/opensubs_all/noblank.numbered.$SRC-$TRG.$SRC; 
	rm $$
fi

exit

# Zip pre-processed files for storage
for lang in $SRC $TRG; do
	echo "Zipping files $lang"
	for file in raw precleaned birecoded cleaned; do
		if [ -f $working_dir/opensubs_all/$typefile.$SRC-$TRG.$lang ]; then
			gzip $working_dir/opensubs_all/$typefile.$SRC-$TRG.$lang
		fi
	done
done








		  
